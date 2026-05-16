import logging
import json


class PrettyJsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'asctime': self.formatTime(record),
            'levelname': record.levelname,
            'name': record.name,
            'filename': record.filename,
            'lineno': record.lineno,
            'message': record.getMessage()
        }
        return json.dumps(log_entry, indent=2)