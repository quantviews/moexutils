-- Роль только для чтения хранилища moexutils (для проектов-потребителей).
--
-- Роль создана этим скриптом 30.09.2026. Повторно выполнять только после новой
-- установки PostgreSQL или восстановления базы moex_lake из копии (скрипт
-- идемпотентен), от суперпользователя PostgreSQL:
--   psql -h localhost -U postgres -d moex_lake -f scripts/reader_role.sql
-- затем задать пароль (вводится интерактивно, в файлы и историю не попадает):
--   psql -h localhost -U postgres -d moex_lake -c "\password moex_reader"
-- и добавить строку в %APPDATA%\postgresql\pgpass.conf пользователя-потребителя:
--   localhost:5432:moex_lake:moex_reader:<пароль>
--
-- Потребитель задает MOEX_PG_USER=moex_reader и читает функциями moexutils
-- (lake.query, stocks.read_*, history.read*) или SQL через DuckDB; запись
-- в хранилище этой ролью невозможна — ее ведет только роль moex.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'moex_reader') THEN
        CREATE ROLE moex_reader LOGIN;
    END IF;
END
$$;

GRANT CONNECT ON DATABASE moex_lake TO moex_reader;
GRANT USAGE ON SCHEMA public TO moex_reader;
-- таблицы каталога DuckLake (метаданные, снимки, список файлов)
GRANT SELECT ON ALL TABLES IN SCHEMA public TO moex_reader;
-- и те, что DuckLake создаст позже (новые версии формата каталога)
ALTER DEFAULT PRIVILEGES FOR ROLE moex IN SCHEMA public GRANT SELECT ON TABLES TO moex_reader;
