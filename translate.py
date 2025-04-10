import os
import re
import time
import json
import argparse
import statistics
from lxml import etree as ET
from langdetect import detect_langs, DetectorFactory
from deep_translator import GoogleTranslator
from functools import lru_cache

DetectorFactory.seed = 0

BB_RE = re.compile(r"(\[/?[^\]]+\])")
PLACEHOLDER_RE = re.compile(r"{[^{}]+}")
CACHE_FILE = "translate_cache.json"

def detect_bbcode(text: str) -> bool:
    return '[' in text and ']' in text

class GoogleTranslateWrapper:
    def __init__(self):
        self.translator = GoogleTranslator(source='en', target='ru')

    def translate(self, text: str) -> str:
        return self.translator.translate(text)

    def translate_batch(self, texts):
        return [self.translate(t) for t in texts]

def mask_placeholders(text: str):
    placeholders = []
    def replace_func(match):
        idx = len(placeholders)
        placeholders.append(match.group(0))
        return f"[[PH{idx}]]"
    masked = re.sub(PLACEHOLDER_RE, replace_func, text)
    return masked, placeholders

def unmask_placeholders(text: str, placeholders):
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
            # Уже структура: {"text": "...", "source": "..."}
            enriched_cache[k] = v
        else:
            enriched_cache[k] = {
                "text": v,
                "source": source_map.get(k, "unknown")
            }
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched_cache, f, ensure_ascii=False, indent=2)

def is_probably_english(text: str, threshold=0.9) -> bool:
    """
    Проверяем, является ли текст с большой вероятностью английским.
    """
    try:
        langs = detect_langs(text)
        for langprob in langs:
            if langprob.lang == 'en' and langprob.prob >= threshold:
                return True
        return False
    except:
        return False

def translate_bbcode_preserving_tags(text: str, translator):
    parts = BB_RE.split(text)
    translated_parts = []
    for part in parts:
        if BB_RE.match(part):
            translated_parts.append(part)  # BB-тег
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

def batch_translate(texts, translator, delay_seconds=1.5, max_chars_per_batch=3000, sources=None):
    """
    Переводим списком (batched). Пишем в локальный кэш.
    Параметр `sources` — список такой же длины, что texts, чтобы сохранить в "source".
    """
    cache = load_cache()
    source_map = {}
    results = []
    to_translate = []
    masked_texts = []
    all_placeholders = []
    original_map = []

    for idx, text in enumerate(texts):
        if isinstance(cache.get(text), dict):
            results.append(cache[text]["text"])
        elif text in cache:
            results.append(cache[text])
        else:
            results.append(None)
            to_translate.append(text)
            masked, placeholders = mask_placeholders(text)
            masked_texts.append(masked)
            all_placeholders.append(placeholders)

            field = sources[idx] if sources and idx < len(sources) else "unknown"
            original_map.append((len(results) - 1, field))

    if not to_translate:
        print("⚡ Всё найдено в кэше.")
        return results

    # Разбиваем на батчи по символам
    batch_queue = []
    batch = []
    batch_len = 0

    for i, masked_text in enumerate(masked_texts):
        text_len = len(masked_text)
        
        # если строка слишком длинная — отправим как отдельный батч
        if text_len > max_chars_per_batch:
            if batch:
                batch_queue.append(batch)
                batch = []
                batch_len = 0
            batch_queue.append([masked_text])
            continue

        if batch_len + text_len > max_chars_per_batch:
            batch_queue.append(batch)
            batch = []
            batch_len = 0

        batch.append(masked_text)
        batch_len += text_len

    if batch:
        batch_queue.append(batch)

    total_batches = len(batch_queue)
    print(f"🔄 Кэшировано: {len(texts) - len(to_translate)}, на перевод: {len(to_translate)}")
    print(f"🔄 Всего батчей: {total_batches} (до {max_chars_per_batch} символов каждый)")

    import statistics
    batch_times = []
    dynamic_delay = delay_seconds
    max_delay = 60
    min_delay = 1.0
    translated_all = []
    start_idx = 0

    for i, batch_part in enumerate(batch_queue, start=1):
        print(f"➡️ Батч {i} из {total_batches}…")
        start_time = time.time()
        success = False
        retries = 0

        while not success and retries < 5:
            try:
                translated = translator.translate_batch(batch_part)
                for j, translated_text in enumerate(translated):
                    placeholders = all_placeholders[start_idx + j]
                    translated[j] = unmask_placeholders(translated_text, placeholders)

                translated_all.extend(translated)

                # Записываем в кэш
                for j, src_text in enumerate(to_translate[start_idx:start_idx+len(translated)]):
                    _, source_field = original_map[start_idx + j]
                    cache[src_text] = translated[j]
                    source_map[src_text] = source_field

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

        if not success:
            raise RuntimeError(f"❌ Перевод батча {i} не удался после 5 попыток.")

        duration = time.time() - start_time
        batch_times.append(duration)

        if success and i < total_batches:
            avg = statistics.mean(batch_times)
            remaining = total_batches - i
            eta = avg * remaining + dynamic_delay * remaining
            print(f"⏳ Осталось примерно: {int(eta)} сек. (ещё {remaining} батчей)")

        start_idx += len(batch_part)
        time.sleep(dynamic_delay)

    for idx, text in enumerate(texts):
        if results[idx] is None:
            if isinstance(cache.get(text), dict):
                results[idx] = cache[text]["text"]
            else:
                results[idx] = cache[text]

    print("✅ Перевод завершён (с кэшем).")
    return results


##################
#   ВАЖНАЯ ЧАСТЬ #
##################

def find_parent_elements(root, parent_tag: str):
    """
    Возвращает все <parent_tag> внутри root (т. е. .//parent_tag),
    плюс, если корень сам называется parent_tag, тоже добавляем root.
    """
    matched = root.findall(f".//{parent_tag}")
    if root.tag == parent_tag:
        matched.insert(0, root)
    return matched

def parse_field_with_condition(field_str: str):
    """
    Разбираем строку вида: "SeqVar_String::Str?UserFacing=True"
    Возвращаем (parent_tag, child_tag, cond_tag, cond_val, is_parent_child).
    """
    condition_tag = None
    condition_val = None
    parent_tag = None
    child_tag = None
    is_parent_child = False

    # Сначала отделяем возможное "?cond=val"
    if "?" in field_str:
        main_part, cond_part = field_str.split("?", 1)
    else:
        main_part = field_str
        cond_part = None

    # Если есть cond_part вроде "UserFacing=True"
    if cond_part and "=" in cond_part:
        condition_tag, condition_val = cond_part.split("=", 1)
        condition_tag = condition_tag.strip()
        condition_val = condition_val.strip()

    # Теперь проверяем, есть ли "::"
    if "::" in main_part:
        parent_tag, child_tag = main_part.split("::", 1)
        parent_tag = parent_tag.strip()
        child_tag = child_tag.strip()
        is_parent_child = True
    else:
        parent_tag = main_part.strip()

    return parent_tag, child_tag, condition_tag, condition_val, is_parent_child

def check_parent_condition(parent_elem: ET.Element, condition_tag: str, condition_val: str) -> bool:
    """
    Если condition_tag не задан, возвращаем True (нет условия).
    Иначе ищем <condition_tag> внутри parent_elem (.find(condition_tag)).
    Сравниваем .text.strip() == condition_val.
    """
    if not condition_tag:
        return True
    found = parent_elem.find(condition_tag)
    if found is not None and found.text and found.text.strip() == condition_val:
        return True
    return False

def translate_fields(root_folder, fields, delay_seconds=1.5, max_chars_per_batch=3000, strict=False):
    """
    Полная реализация:
    - fields может содержать как "Description" (одиночный тег),
      так и "SeqVar_String::Str?UserFacing=True" (родитель::ребёнок + условие).
    - Если есть условие "?UserFacing=True", то перед тем, как переводить <Str>,
      проверяем внутри родителя <UserFacing> со значением "True".
    """
    translator = GoogleTranslateWrapper()
    parser = ET.XMLParser(remove_blank_text=True, recover=not strict)

    texts_to_translate = []
    sources_for_translate = []
    text_locations = []
    modified_files = {}
    error_log = []

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.lower().endswith('.xml'):
                continue
            file_path = os.path.join(dirpath, filename)

            try:
                tree = ET.parse(file_path, parser)
                root = tree.getroot()
                modified_files[file_path] = (tree, False)

                # Для каждого поля-условия
                for field_def in fields:
                    parent_tag, child_tag, cond_tag, cond_val, is_parent_child = parse_field_with_condition(field_def)

                    if is_parent_child:
                        # Сначала ищем всех parent_tag
                        parent_elems = find_parent_elements(root, parent_tag)
                        for p_elem in parent_elems:
                            # Проверяем условие (cond_tag=UserFacing? cond_val=?)
                            if not check_parent_condition(p_elem, cond_tag, cond_val):
                                continue
                            # Если условие проходит, ищем внутри p_elem -> child_tag
                            child_elems = p_elem.findall(f".//{child_tag}")
                            for elem in child_elems:
                                if not elem.text:
                                    continue
                                txt = elem.text.strip()
                                if is_probably_english(txt):
                                    # Собираем
                                    texts_to_translate.append(txt)
                                    # Для кэша: пишем полный путь + название полей
                                    src_info = f"{file_path}:{parent_tag}::{child_tag}"
                                    if cond_tag and cond_val:
                                        src_info += f"?{cond_tag}={cond_val}"
                                    sources_for_translate.append(src_info)
                                    text_locations.append((elem, file_path))
                    else:
                        # Одиночный тег
                        # Сначала ищем .//parent_tag
                        found_elems = root.findall(f".//{parent_tag}")
                        if root.tag == parent_tag:
                            found_elems.insert(0, root)
                        for elem in found_elems:
                            # проверяем условие к самому elem
                            if not check_parent_condition(elem, cond_tag, cond_val):
                                continue
                            if not elem.text:
                                continue
                            txt = elem.text.strip()
                            if is_probably_english(txt):
                                texts_to_translate.append(txt)
                                src_info = f"{file_path}:{parent_tag}"
                                if cond_tag and cond_val:
                                    src_info += f"?{cond_tag}={cond_val}"
                                sources_for_translate.append(src_info)
                                text_locations.append((elem, file_path))

            except Exception as e:
                msg = f"[!] Ошибка чтения {file_path}: {e}"
                print(msg)
                error_log.append(msg)

    if not texts_to_translate:
        print("🎉 Всё уже переведено или ничего не найдено.")
        return

    print(f"🔍 Найдено строк для перевода: {len(texts_to_translate)}")
    translated_texts = batch_translate(
        texts_to_translate,
        translator,
        delay_seconds=delay_seconds,
        max_chars_per_batch=max_chars_per_batch,
        sources=sources_for_translate
    )

    # Раскладываем переводы обратно
    for (elem, file_path), new_text in zip(text_locations, translated_texts):
        # Если у вас bbcode, можно сделать доп. проверку:
        # if detect_bbcode(elem.text or ""):
        #     elem.text = translate_bbcode_preserving_tags(elem.text, translator)
        # else:
        #     elem.text = new_text
        elem.text = new_text
        modified_files[file_path] = (modified_files[file_path][0], True)

    # Сохраняем
    for file_path, (tree, was_modified) in modified_files.items():
        if was_modified:
            try:
                tree.write(file_path, encoding='utf-8', xml_declaration=True, pretty_print=True)
                print(f"✅ Сохранено: {file_path}")
            except Exception as e:
                msg = f"[!] Ошибка сохранения {file_path}: {e}"
                print(msg)
                error_log.append(msg)

    if error_log:
        with open("log_errors.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(error_log))
        print("\n⚠️ Ошибки записаны в log_errors.txt")

def main():
    parser = argparse.ArgumentParser(description="Перевод XML (с поддержкой Parent::Child?Tag=Value)")
    parser.add_argument("--path", required=True, help="Папка с .xml файлами")
    parser.add_argument("--fields", nargs="+", default=["SeqVar_String::Str?UserFacing=True"],
                        help="Поля для перевода, напр: SeqVar_String::Str?UserFacing=True Description")
    parser.add_argument("--strict", action="store_true", help="Отключить recover в XML-парсинге")
    parser.add_argument("--delay", type=float, default=1.5, help="Задержка между батчами")
    parser.add_argument("--maxchars", type=int, default=3000, help="Макс. символов в батче")
    args = parser.parse_args()

    translate_fields(
        root_folder=args.path,
        fields=args.fields,
        delay_seconds=args.delay,
        max_chars_per_batch=args.maxchars,
        strict=args.strict
    )

if __name__ == "__main__":
    main()
