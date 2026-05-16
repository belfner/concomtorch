#!/usr/bin/env python3
"""
Tiny POST-to-ntfy helper. Reads URL from CONCOMTORCH_NTFY_URL env var; no-ops when unset.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request

from loguru import logger

from logging_setup import setup_logging


def notify(message: str, title: str | None = None, priority: str = 'default') -> bool:
    """
    Send a notification via ntfy.

    Parameters
    ----------
    message : str
        Body of the notification.
    title : str, optional
        Notification title.
    priority : str
        ntfy priority: min/low/default/high/urgent.

    Returns
    -------
    bool
        True when the HTTP POST succeeded, False when no URL is configured or the request failed.
    """
    url = os.environ.get('CONCOMTORCH_NTFY_URL', '').strip()
    if url == '':
        logger.warning('CONCOMTORCH_NTFY_URL not set; skipping notification.')
        return False
    headers = {'Priority': priority}
    if title is not None:
        headers['Title'] = title
    req = urllib.request.Request(url, data=message.encode('utf-8'), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError) as exc:
        logger.error(f'ntfy POST failed: {exc}')
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--title', default=None)
    parser.add_argument('--priority', default='default',
                        choices=['min', 'low', 'default', 'high', 'urgent'])
    parser.add_argument('message')
    args = parser.parse_args()
    setup_logging('notify')
    ok = notify(args.message, title=args.title, priority=args.priority)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
