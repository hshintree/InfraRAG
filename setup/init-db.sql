CREATE USER infrarag_user WITH PASSWORD 'infrarag_secure_pw_2024';
CREATE DATABASE infrarag_db OWNER infrarag_user;
GRANT ALL PRIVILEGES ON DATABASE infrarag_db TO infrarag_user;

\c infrarag_db;
CREATE EXTENSION IF NOT EXISTS vector;
GRANT ALL ON SCHEMA public TO infrarag_user;
