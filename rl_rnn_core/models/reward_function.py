from enum import Enum
from datetime import datetime as dt
from ..service import db_manager as dbm
from .base_models import BaseModelsClass as BCM


class TrainingStatus(Enum):
    """Status for training lifecycle."""
    PLANNED = 'planned'
    TESTED = 'tested'
    TRAINED = 'trained'


class Training_Model(BCM):
    """
    Represents a machine learning training session.

    Maps model, function and process information to training metadata.
    """

    table_name = 'training'
    fields = ["status", "creation_date", "function_id", "process_id", "model_id", "log_path", "best_result", "notes", "name"]

    DB_SCHEMA = '''CREATE TABLE IF NOT EXISTS training (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        status TEXT,
        creation_date TEXT,
        function_id INTEGER,
        process_id INTEGER,
        model_id INTEGER, 
        log_path TEXT,
        best_result REAL,
        notes TEXT,
        name TEXT,
        FOREIGN KEY (function_id) REFERENCES functions(id),
        FOREIGN KEY (model_id) REFERENCES models(id),
        FOREIGN KEY (process_id) REFERENCES processes(id)
    );'''

    INSERT_QUERY = '''INSERT INTO training (status, creation_date, function_id, process_id, model_id, log_path, best_result, notes, name)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);'''

    def __init__(self, **kwargs):
        # Default creation date if not passed
        if not kwargs.get("creation_date"):
            kwargs["creation_date"] = dt.now().strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(kwargs.get("status"), TrainingStatus):
            kwargs["status"] = kwargs["status"].value
        super().__init__(**kwargs)

    def to_tuple(self):
        """Serialize the instance fields to a tuple."""
        return (
            self.status,
            self.creation_date,
            self.function_id,
            self.process_id,
            self.model_id,
            self.log_path,
            self.best_result,
            self.notes,
            self.name
        )

    @classmethod
    def convert_db_response(cls, row):
        """
        Convert a DB row into a Training_Model instance.

        Args:
            row (tuple): DB row in the format (id, status, creation_date, ...)

        Returns:
            Training_Model: Instance from database row
        """
        try:
            data = dict(zip(("id", *cls.fields), row))
            data["status"] = TrainingStatus(data["status"]).value
            return cls(**data)
        except Exception as e:
            raise ValueError(f"Error converting DB row to Training_Model: {e}")

    def update_status(self, status: TrainingStatus):
        """Update the training status."""
        dbm.new_update_record(self.table_name, {'status': status.value}, {'id': self.id})

    def update_best_result(self, best_result: float):
        """Update the best result if it improves the current one."""
        try:
            if float(self.best_result) < best_result:
                dbm.new_update_record(self.table_name, {'best_result': best_result}, {'id': self.id})
        except:
            dbm.new_update_record(self.table_name, {'best_result': best_result}, {'id': self.id})

    def update_path(self, path: str):
        """Update the log file path."""
        dbm.new_update_record(self.table_name, {'log_path': path}, {'id': self.id})

    def update_notes(self, notes: str):
        """Update notes field."""
        dbm.new_update_record(self.table_name, {'notes': notes}, {'id': self.id})

    @staticmethod
    def retrieve_list_records_by_name(names: list[str]):
        """Retrieve a list of training records by name."""
        return dbm.retive_a_list_of_recordos('name', 'training', names)
