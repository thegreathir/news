import argparse
import os
import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader, select_autoescape
from openai import OpenAI


# ==============================
# CONFIG
# ==============================

CHANNELS = [
    "khabar_fouri",
    "iranintltv",
    "vahidonline",
    "farsna",
]

HOURS_BACK = 6
TARGET_CHAT = "@sarkhattekhabarha"

SLEEP_BETWEEN_PAGES_SEC = 1.0
MAX_PAGES_PER_CHANNEL = 250
REQUEST_TIMEOUT_SEC = 30

MODEL_NAME = "gpt-5.4"
BASE_URL = "https://t.me/s"


# ==============================
# Logging
# ==============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

MESSAGES_DUMP_PATH = "messages_dump.html"
MESSAGES_TEMPLATE_PATH = Path(__file__).with_name("messages_dump_template.html")


# ==============================
# Scraper
# ==============================


def _parse_datetime(dt_str: str) -> datetime:
    s = dt_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _isoformat_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_host_local_timestamp(dt_str: str) -> Dict[str, str]:
    local_dt = _parse_datetime(dt_str).astimezone()
    timezone_label = local_dt.tzname() or "Local"
    return {
        "datetime": local_dt.isoformat(timespec="seconds"),
        "date_label": local_dt.strftime("%Y-%m-%d"),
        "time_label": local_dt.strftime("%H:%M:%S"),
        "timezone_label": timezone_label,
    }


def _clean_text(node) -> str:
    if not node:
        return ""
    for br in node.find_all("br"):
        br.replace_with("\n")
    text = node.get_text(separator="", strip=False).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text


def _fetch(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=REQUEST_TIMEOUT_SEC)
    r.raise_for_status()
    return r.text


def _extract_messages(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    wraps = soup.select("div.tgme_widget_message_wrap")

    out = []
    for w in wraps:
        msg_div = w.select_one("div.tgme_widget_message")
        if not msg_div or not msg_div.has_attr("data-post"):
            continue

        m = re.search(r"/(\d+)$", msg_div["data-post"])
        if not m:
            continue

        msg_id = int(m.group(1))

        # Some posts include media-duration <time> elements before the message date.
        # Only parse the date element that carries a datetime attribute.
        time_tag = w.select_one(
            "a.tgme_widget_message_date time[datetime]"
        ) or w.select_one("time[datetime]")
        if not time_tag or not time_tag.has_attr("datetime"):
            continue

        dt = _parse_datetime(time_tag["datetime"])
        # Telegram uses `tgme_widget_message_text` for both reply previews and
        # the actual post body. Restrict to `.js-message_text` so reply posts
        # don't get their quoted preview extracted as the message body.
        text_div = w.select_one("div.tgme_widget_message_text.js-message_text")
        text = _clean_text(text_div)

        out.append({"msg_id": msg_id, "dt": dt, "text": text})

    return out


def scrape_channel(channel: str, cutoff: datetime) -> List[Dict[str, str]]:
    logger.info(f"Scraping channel: {channel}")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    results = []
    seen_ids = set()
    next_before = None
    pages = 0
    stop = False

    while not stop and pages < MAX_PAGES_PER_CHANNEL:
        pages += 1
        url = (
            f"{BASE_URL}/{channel}"
            if not next_before
            else f"{BASE_URL}/{channel}?before={next_before}"
        )

        html = _fetch(session, url)
        msgs = _extract_messages(html)
        if not msgs:
            break

        oldest_dt_on_page = None
        msgs_sorted = sorted(msgs, key=lambda x: x["msg_id"], reverse=True)

        for m in msgs_sorted:
            msg_id = m["msg_id"]
            dt = m["dt"]
            text = m["text"]

            if oldest_dt_on_page is None or dt < oldest_dt_on_page:
                oldest_dt_on_page = dt

            if msg_id in seen_ids:
                continue
            seen_ids.add(msg_id)

            if dt >= cutoff and text:
                results.append(
                    {
                        "msg": text,
                        "datetime": _isoformat_z(dt),
                        "source": channel,
                    }
                )

        if oldest_dt_on_page and oldest_dt_on_page < cutoff:
            stop = True

        next_before = min(m["msg_id"] for m in msgs)
        time.sleep(SLEEP_BETWEEN_PAGES_SEC)

    logger.info(f"{channel}: collected {len(results)} messages")
    return results


# ==============================
# OpenAI Digest
# ==============================


def generate_digest(messages: List[Dict[str, str]]) -> str:
    logger.info("Generating Persian digest using GPT-5.2")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages_json = json.dumps(messages, ensure_ascii=False)

    prompt = f"""
نقش: تحلیلگر ارشد خبر فارسی برای خلاصه‌سازی محتوای تلگرام.

ورودی: آرایه JSON از پست‌ها با فیلدهای msg, datetime, source.
قواعد:
1) فقط بر اساس داده ورودی بنویس؛ هیچ خبر یا تحلیل بیرونی اضافه نکن.
2) موارد تکراری را ادغام کن. اگر چند پست درباره یک رویدادند، یک مورد نهایی بساز و همه منابع مرتبط را ذکر کن.
3) اخبار را بر اساس اهمیت مرتب کن: اثر سیاسی/امنیتی/اقتصادی > تازگی > تعداد منابع.
4) اگر داده کافی نیست، با برچسب «نامطمئن» مشخص کن.
5) خروجی کاملا فارسی و بدون Markdown باشد (بدون # * - ```).
6) متن مناسب تلگرام، روان، فشرده، حدود 600 تا 900 کلمه.
7) برای هر خبر منبع/منابع را ذکر کن.

فرمت خروجی دقیق:
🧩 جمع‌بندی کوتاه
(۳ تا ۵ جمله از مهم‌ترین وضعیت کلی)

🔥 مهم‌ترین رویدادها
(۵ تا ۸ مورد؛ هر مورد شامل: عنوان کوتاه + شرح ۲-۳ جمله + منبع/منابع)

📌 نکات کلیدی و الگوها
(۳ تا 6 نکته تحلیلی از روندها و ارتباط خبرها)

✅ جمع‌بندی نهایی
(۲ تا ۴ جمله با تصویر کلی و ریسک‌های احتمالی نزدیک)

فقط متن نهایی را برگردان.

JSON:
{messages_json}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
    )

    return response.output_text.strip()


# ==============================
# Telegram Bot Sender
# ==============================


def _chunk_text_safely(text: str, limit: int = 3900) -> List[str]:
    if len(text) <= limit:
        return [text]

    chunks = []
    current = ""
    parts = text.split("\n\n")

    for part in parts:
        candidate = part if not current else f"{current}\n\n{part}"
        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(part) > limit:
            lines = part.split("\n")
            line_buf = ""

            for line in lines:
                candidate_line = line if not line_buf else f"{line_buf}\n{line}"
                if len(candidate_line) <= limit:
                    line_buf = candidate_line
                    continue

                if line_buf:
                    chunks.append(line_buf)
                line_buf = line

                # Fallback if a single line is still too long.
                while len(line_buf) > limit:
                    split_at = line_buf.rfind(" ", 0, limit)
                    if split_at == -1:
                        split_at = limit
                    chunks.append(line_buf[:split_at].rstrip())
                    line_buf = line_buf[split_at:].lstrip()

            if line_buf:
                current = line_buf
        else:
            current = part

    if current:
        chunks.append(current)

    if len(chunks) > 1:
        total = len(chunks)
        chunks = [f"({idx}/{total})\n{chunk}" for idx, chunk in enumerate(chunks, 1)]

    return chunks


def send_via_bot(text: str):
    logger.info(f"Sending digest to {TARGET_CHAT}")

    bot_token = os.environ["BOT_TOKEN"]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    chunks = _chunk_text_safely(text, limit=3900)

    for idx, chunk in enumerate(chunks, start=1):
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id": TARGET_CHAT,
                    "text": chunk,
                    "disable_web_page_preview": True,
                },
                timeout=20,
            )

            if resp.status_code != 200:
                logger.error("Telegram API returned non-200 response")
                logger.error(f"Status Code: {resp.status_code}")
                logger.error(f"Raw Response: {resp.text}")

                try:
                    error_json = resp.json()
                    logger.error(f"Telegram error_code: {error_json.get('error_code')}")
                    logger.error(
                        f"Telegram description: {error_json.get('description')}"
                    )
                except Exception:
                    logger.error("Failed to parse Telegram error response as JSON")

                resp.raise_for_status()

        except requests.exceptions.RequestException as e:
            logger.exception(
                "Network or request exception occurred while sending message"
            )
            raise e

    logger.info("Digest successfully sent")


def dump_messages(
    messages: List[Dict[str, str]], path: str = MESSAGES_DUMP_PATH
) -> None:
    logger.info(f"Writing collected messages to {path}")

    channels = sorted({message["source"] for message in messages})
    channel_views = []

    for index, channel in enumerate(channels):
        grouped_messages = sorted(
            [message for message in messages if message["source"] == channel],
            key=lambda item: item["datetime"],
        )
        rendered_messages = [
            {
                **message,
                **_format_host_local_timestamp(message["datetime"]),
            }
            for message in grouped_messages
        ]
        channel_views.append(
            {
                "name": channel,
                "is_active": index == 0,
                "tab_id": f"tab-{channel}",
                "panel_id": f"panel-{channel}",
                "messages": rendered_messages,
                "message_count": len(rendered_messages),
            }
        )

    template_env = Environment(
        loader=FileSystemLoader(str(MESSAGES_TEMPLATE_PATH.parent)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = template_env.get_template(MESSAGES_TEMPLATE_PATH.name)
    html_document = template.render(channels=channel_views)

    with open(path, "w", encoding="utf-8") as f:
        f.write(html_document)


# ==============================
# Main
# ==============================


def collect_recent_messages(hours_back: int = HOURS_BACK) -> List[Dict[str, str]]:
    logger.info(f"Starting aggregation for last {hours_back} hours")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    all_messages = []

    for ch in CHANNELS:
        try:
            all_messages.extend(scrape_channel(ch, cutoff))
        except Exception:
            logger.exception(f"Error scraping channel {ch}")

    logger.info(f"Total messages collected: {len(all_messages)}")

    if not all_messages:
        logger.info("No recent messages found.")
        return []

    all_messages.sort(key=lambda x: (x["datetime"], x["source"]))
    return all_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Telegram channels, generate a digest, or dump messages to HTML."
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Write scraped messages to a single HTML file instead of generating and sending the digest.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_messages = collect_recent_messages()

    if not all_messages:
        return

    if args.dump:
        dump_messages(all_messages)
        logger.info("Process completed successfully")
        return

    digest = generate_digest(all_messages)

    # DO NOT log digest content
    send_via_bot(digest)

    logger.info("Process completed successfully")


if __name__ == "__main__":
    main()
