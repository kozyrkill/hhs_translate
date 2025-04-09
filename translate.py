import os
import re
import time
import argparse
import statistics
from lxml import etree as ET
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # стабильность

CYRILLIC_RE = re.compile('[а-яА-Я]')

def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except Exception:
        return False

def batch_translate(texts, translator, delay_seconds=1.5, max_chars_per_batch=3000):
    results = []
    batch = []
    batch_queue = []
    batch_len = 0

    for text in texts:
        if batch_len + len(text) > max_chars_per_batch and batch:
            batch_queue.append(batch)
            batch = []
            batch_len = 0
        batch.append(text)
        batch_len += len(text)
    if batch:
        batch_queue.append(batch)

    total_batches = len(batch_queue)
    print(f"🔄 Всего батчей: {total_batches} (до {max_chars_per_batch} символов каждый)")

    batch_times = []

    for i, batch in enumerate(batch_queue, 1):
        print(f"➡️ Батч {i} из {total_batches}…")
        start_time = time.time()
        try:
            translated = translator.translate_batch(batch)
            results.extend(translated)
        except Exception as e:
            print(f'[!] Ошибка при переводе батча {i}: {e}')
            results.extend(batch)

        duration = time.time() - start_time
        batch_times.append(duration)
        time.sleep(delay_seconds)

        if i < total_batches:
            avg = statistics.mean(batch_times)
            remaining = total_batches - i
            eta = avg * remaining + delay_seconds * remaining
            print(f"⏳ Осталось примерно: {int(eta)} сек. ({remaining} батчей)")

    print("✅ Перевод завершён.")
    return results

def translate_fields(root_folder, fields, delay_seconds=1.5, max_chars_per_batch=3000, strict=False):
    translator = GoogleTranslator(source='en', target='ru')
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
                    if localname in fields and elem.text:
                        text = elem.text.strip()
                        if is_english(text):
                            to_translate.append(text)
                            text_locations.append((file_path, elem))

            except Exception as e:
                msg = f'[!] Ошибка чтения {file_path}: {e}'
                print(msg)
                error_log.append(msg)

    print(f'🔍 Строк на перевод: {len(to_translate)} из полей: {fields}')
    if not to_translate:
        print("🎉 Всё уже переведено.")
    else:
        translated_texts = batch_translate(to_translate, translator, delay_seconds, max_chars_per_batch)

        for (file_path, elem), translated in zip(text_locations, translated_texts):
            elem.text = translated
            if file_path in modified_files:
                tree, _ = modified_files[file_path]
                modified_files[file_path] = (tree, True)

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
    parser = argparse.ArgumentParser(description="Перевод XML-тегов с поддержкой namespace и проверкой языка.")
    parser.add_argument("--path", required=True, help="Путь к папке с XML-файлами")
    parser.add_argument("--fields", nargs="+", default=["Description Str"], help="Список тегов (по умолчанию: Description)")
    parser.add_argument("--strict", action="store_true", help="Отключить режим recover и валидировать XML строго")
    args = parser.parse_args()

    translate_fields(
        args.path,
        args.fields,
        strict=args.strict
    )
