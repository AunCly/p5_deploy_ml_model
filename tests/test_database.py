import sys
from unittest.mock import patch, MagicMock, mock_open

class TestDatabaseManager:
    def test_get_engine(self):
        with patch('database.database_manager.create_engine') as mock_create_engine:
            from database import database_manager
            database_manager.get_engine()
            mock_create_engine.assert_called_once()

    def test_create_database_success(self):
        mock_engine = MagicMock()
        mock_conn = mock_engine.begin.return_value.__enter__.return_value
        with patch('database.database_manager.get_engine', return_value=mock_engine):
            with patch('builtins.open', mock_open(read_data='CREATE TABLE test;')):
                from database import database_manager
                database_manager.create_database()
                mock_conn.execute.assert_called_once()

    def test_create_database_file_not_found(self, capsys):
        with patch('database.database_manager.get_engine'):
            with patch('builtins.open', side_effect=FileNotFoundError):
                from database import database_manager
                database_manager.create_database()
                captured = capsys.readouterr()
                assert 'introuvable' in captured.out

    def test_create_database_exception(self, capsys):
        mock_engine = MagicMock()
        mock_conn = mock_engine.begin.return_value.__enter__.return_value
        mock_conn.execute.side_effect = Exception("DB error")
        with patch('database.database_manager.get_engine', return_value=mock_engine):
            with patch('builtins.open', mock_open(read_data='SQL')):
                from database import database_manager
                database_manager.create_database()
                captured = capsys.readouterr()
                assert 'erreur' in captured.out

    def test_save_prediction(self):
        mock_engine = MagicMock()
        mock_conn = mock_engine.begin.return_value.__enter__.return_value
        with patch('database.database_manager.get_engine', return_value=mock_engine):
            from database import database_manager
            database_manager.save_prediction('{"id": 1}', {'prediction': 1, 'probability': 0.8})
            mock_conn.execute.assert_called_once()

    def test_get_predictions(self):
        mock_engine = MagicMock()
        mock_conn = mock_engine.begin.return_value.__enter__.return_value
        mock_conn.execute.return_value.mappings.return_value = [{'id': 1, 'prediction': 1}]
        with patch('database.database_manager.get_engine', return_value=mock_engine):
            from database import database_manager
            result = database_manager.get_predictions()
            assert result == [{'id': 1, 'prediction': 1}]


class TestSeeding:
    def test_seed(self):
        mock_df = MagicMock()
        mock_engine = MagicMock()
        with patch('database.seeding.pd.read_csv', return_value=mock_df):
            with patch('database.seeding.database.get_engine', return_value=mock_engine):
                from database import seeding
                seeding.seed()
                assert mock_df.to_sql.call_count == 3


class TestInitDatabase:
    def test_init_database(self):
        with patch('database.database_manager.create_database') as mock_create:
            with patch('database.seeding.seed') as mock_seed:
                if 'database.init_database' in sys.modules:
                    del sys.modules['database.init_database']
                import database.init_database
                mock_create.assert_called_once()
                mock_seed.assert_called_once()
