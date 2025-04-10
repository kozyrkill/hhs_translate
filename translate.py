import os
import re
import time
import json
import argparse
import statistics
from lxml import etree as ET
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from functools import lru_cache

DetectorFactory.seed = 0

CYRILLIC_RE = re.compile('[а-яА-Я]')
BB_RE = re.compile(r"(\[/?[^\]]+\])")
PLACEHOLDER_RE = re.compile(r"{[^{}]+}")
CACHE_FILE = "translate_cache.json"

class GoogleTranslateWrapper:
    def __init__(self):
        self.translator = GoogleTranslator(source='en', target='ru')

    def translate(self, text):
        return self.translator.translate(text)

    def translate_batch(self, texts):
        return [self.translate(text) for text in texts]

def mask_placeholders(text):
    placeholders = PLACEHOLDER_RE.findall(text)
    masked = text
    for i, ph in enumerate(placeholders):
        masked = masked.replace(ph, f"[[PH{i}]]")
    return masked, placeholders

def unmask_placeholders(text, placeholders):
    for i, ph in enumerate(placeholders):
        pattern = re.compile(re.escape(f"[[PH{i}]]"), re.IGNORECASE)
        text = pattern.sub(ph, text)
    return text

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Повреждённый кэш — игнорируем translate_cache.json")
            return {}
    return {}

def save_cache_with_source(cache, source_map):
    enriched_cache = {}
    for k, v in cache.items():
        if isinstance(v, dict):
            enriched_cache[k] = v
        else:
            enriched_cache[k] = {
                "text": v,
                "source": source_map.get(k, "unknown")
            }
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched_cache, f, ensure_ascii=False, indent=2)

def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except Exception:
        return False

@lru_cache(maxsize=10000)
def translate_bbcode_preserving_tags(text, translator):
    parts = BB_RE.split(text)
    translated_parts = []
    for part in parts:
        if BB_RE.match(part):
            translated_parts.append(part)
        else:
            lines = part.split('\n')
            translated_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        masked, placeholders = mask_placeholders(line)
                        translated_line = translator.translate(masked) or ''
                        translated_line = unmask_placeholders(translated_line, placeholders)
                        translated_lines.append(translated_line)
                    except Exception as e:
                        print(f"[!] Ошибка перевода BB-кода строки: {line} — {e}")
                        translated_lines.append(line)
                else:
                    translated_lines.append('')
            translated_parts.append('\n'.join(translated_lines))
    return ''.join(translated_parts)

def batch_translate(texts, translator, delay_seconds=1.5, max_chars_per_batch=3000, fields=None):
    cache = load_cache()
    source_map = {}
    results = []
    to_translate = []
    original_map = []
    masked_texts = []
    all_placeholders = []

    for idx, text in enumerate(texts):
        if isinstance(cache.get(text), dict):
            results.append(cache[text]["text"])
        elif text in cache:
            results.append(cache[text])
        else:
            masked, placeholders = mask_placeholders(text)
            masked_texts.append(masked)
            all_placeholders.append(placeholders)
            results.append(None)
            to_translate.append(text)
            field = fields[idx] if fields and idx < len(fields) else "unknown"
            original_map.append((len(results) - 1, field))

    if not to_translate:
        print("⚡ Всё найдено в кэше.")
        return results

    batch = []
    batch_queue = []
    batch_len = 0

    for text in masked_texts:
        if batch_len + len(text) > max_chars_per_batch and batch:
            batch_queue.append(batch)
            batch = []
            batch_len = 0
        batch.append(text)
        batch_len += len(text)
    if batch:
        batch_queue.append(batch)

    total_batches = len(batch_queue)
    print(f"🔄 Кэшировано: {len(texts) - len(to_translate)}, на перевод: {len(to_translate)}")
    print(f"🔄 Всего батчей: {total_batches} (до {max_chars_per_batch} символов каждый)")

    batch_times = []
    dynamic_delay = delay_seconds
    max_delay = 60
    min_delay = 1.0
    translated_all = []

    for i, batch in enumerate(batch_queue, 1):
        print(f"➡️ Батч {i} из {total_batches}…")
        start_time = time.time()
        success = False
        retries = 0

        while not success and retries < 5:
            try:
                translated = translator.translate_batch(batch)
                start_index = len(translated_all)
                for j, translated_text in enumerate(translated):
                    placeholders = all_placeholders[start_index + j]
                    translated[j] = unmask_placeholders(translated_text, placeholders)
                translated_all.extend(translated)
                for j, (src, tgt) in enumerate(zip(to_translate[start_index:start_index+len(translated)], translated)):
                    cache[src] = tgt
                    _, source_field = original_map[start_index + j]
                    source_map[src] = source_field
                save_cache_with_source(cache, source_map)
                success = True
                dynamic_delay = max(min_delay, dynamic_delay * 0.9)
            except Exception as e:
                print(f"[!] Ошибка при переводе батча {i}: {e}")
                retries += 1
                dynamic_delay = min(max_delay, dynamic_delay * 1.5)
                wait_time = int(dynamic_delay)
                print(f"⏳ Ждём {wait_time} сек. перед повтором (попытка {retries})…")
                time.sleep(wait_time)

        duration = time.time() - start_time
        batch_times.append(duration)

        if success and i < total_batches:
            avg = statistics.mean(batch_times)
            remaining = total_batches - i
            eta = avg * remaining + dynamic_delay * remaining
            print(f"⏳ Осталось примерно: {int(eta)} сек. ({remaining} батчей)")

        time.sleep(dynamic_delay)

    for idx, _ in original_map:
        results[idx] = cache[texts[idx]] if isinstance(cache[texts[idx]], str) else cache[texts[idx]]["text"]

    print("✅ Перевод завершён (с кэшированием).")
    return results

def translate_fields(root_folder, fields, delay_seconds=1.5, max_chars_per_batch=3000, strict=False):
    translator = GoogleTranslateWrapper()
    to_translate = []
    text_locations = []
    modified_files = {}
    error_log = []

    parser = ET.XMLParser(remove_blank_text=True, recover=not strict)

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.lower().endswith('.xml'):
                continue
            file_path = os.path.join(dirpath, filename)

            try:
                tree = ET.parse(file_path, parser)
                root = tree.getroot()
                modified_files[file_path] = (tree, False)

                for elem in root.iter():
                    localname = ET.QName(elem).localname
                    if localname in fields:
                        if localname.lower() == "decisiontext":
                            for child in elem.findall(".//string"):
                                if child.text and is_english(child.text.strip()):
                                    to_translate.append(child.text.strip())
                                    text_locations.append((file_path, child, 'plain', None))
                        elif elem.text:
                            text = elem.text.strip()
                            if is_english(text):
                                if '[' in text and ']' in text:
                                    text_locations.append((file_path, elem, 'bbcode', None))
                                else:
                                    to_translate.append(text)
                                    text_locations.append((file_path, elem, 'plain', None))

            except Exception as e:
                msg = f'[!] Ошибка чтения {file_path}: {e}'
                print(msg)
                error_log.append(msg)

    print(f'🔍 Строк на перевод: {len(to_translate)} из полей: {fields}')
    if not to_translate:
        print("🎉 Всё уже переведено.")
    else:
        translated_texts = batch_translate(to_translate, translator, delay_seconds, max_chars_per_batch)

        ti = 0
        for (file_path, elem, mode, _), translated in zip(text_locations, translated_texts):
            if mode == 'bbcode':
                elem.text = translate_bbcode_preserving_tags(elem.text, translator)
                modified_files[file_path] = (modified_files[file_path][0], True)
            elif mode == 'plain':
                elem.text = translated
                modified_files[file_path] = (modified_files[file_path][0], True)

        for file_path, (tree, was_modified) in modified_files.items():
            if was_modified:
                try:
                    tree.write(file_path, encoding='utf-8', xml_declaration=True, pretty_print=True)
                    print(f'✅ Сохранено: {file_path}')
                except Exception as e:
                    msg = f'[!] Ошибка сохранения {file_path}: {e}'
                    print(msg)
                    error_log.append(msg)

    if error_log:
        with open("log_errors.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(error_log))
        print(f"\n⚠️ Ошибки записаны в log_errors.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Перевод XML с кэшем, BB-кодами и Google Translate.")
    parser.add_argument("--path", required=True, help="Путь к папке с XML-файлами")
    parser.add_argument("--fields", nargs="+", default=["Description"], help="Список тегов для перевода")
    parser.add_argument("--strict", action="store_true", help="Отключить recover в XML-парсинге")
    args = parser.parse_args()

    translate_fields(
        args.path,
        args.fields,
        strict=args.strict
    )
