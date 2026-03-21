#!/usr/bin/env python3
"""Message logger for recording agent communication to JSONL + text files."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


class MessageLogger:

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), 'logs')

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'agent_messages_{timestamp}.jsonl'
        self.text_log_file = self.log_dir / f'agent_messages_{timestamp}.txt'

        print(f'Message logger — JSONL: {self.log_file}')
        print(f'Message logger — Text:  {self.text_log_file}')

    def log_message(self, sender: str, receiver: str,
                    message_type: str, content: dict):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'sender': sender,
            'receiver': receiver,
            'message_type': message_type,
            'content': content,
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        with open(self.text_log_file, 'a') as f:
            f.write(self._format_text(entry) + '\n')

    def _format_text(self, entry: dict) -> str:
        ts = datetime.fromisoformat(entry['timestamp']).strftime('%H:%M:%S')
        sender = entry['sender']
        receiver = entry['receiver']
        msg_type = entry['message_type']
        c = entry['content']

        if msg_type == 'MoveCommand':
            return (f'[{ts}] {sender} -> {receiver}: '
                    f"Move to {c.get('position', '?')}, "
                    f"priority {c.get('priority', '?')}")
        if msg_type == 'StatusUpdate':
            return (f'[{ts}] {sender} -> {receiver}: '
                    f"{c.get('state', '?')} | "
                    f"water {c.get('water_level', '?')}% | "
                    f"pos {c.get('position', '?')}")
        if msg_type == 'RefillCommand':
            return f'[{ts}] {sender} -> {receiver}: Refill at base'
        return f'[{ts}] {sender} -> {receiver}: {msg_type} — {c}'

    def load_logs(self, log_file: str = None) -> list:
        if log_file is None:
            files = sorted(self.log_dir.glob('agent_messages_*.jsonl'))
            if not files:
                return []
            log_file = files[-1]

        entries = []
        with open(log_file) as f:
            for line in f:
                entries.append(json.loads(line))
        return entries

    def print_analysis(self, log_file: str = None):
        entries = self.load_logs(log_file)
        if not entries:
            print('No logs found')
            return

        counts = {}
        for e in entries:
            t = e['message_type']
            counts[t] = counts.get(t, 0) + 1

        start = datetime.fromisoformat(entries[0]['timestamp'])
        end = datetime.fromisoformat(entries[-1]['timestamp'])
        duration = (end - start).total_seconds()

        print(f'\n=== Communication Analysis ===')
        print(f'Messages: {len(entries)}  Duration: {duration:.1f}s')
        for t, c in counts.items():
            print(f'  {t}: {c}')


if __name__ == '__main__':
    logger = MessageLogger()
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        logger.print_analysis()
    else:
        print(f'Logs directory: {logger.log_dir}')
