-- PostgreSQL Dump converted from MySQL/phpMyAdmin
BEGIN;

-- --------------------------------------------------------
-- Table structure for table "users"
-- --------------------------------------------------------

CREATE TABLE "users" (
  "uid" BIGSERIAL PRIMARY KEY,
  "name" varchar(100) NOT NULL,
  "email" varchar(100) NOT NULL UNIQUE,
  "password" varchar(100) NOT NULL,
  "register_time" timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "user_type" varchar(25) NOT NULL,
  "user_image" text NOT NULL,
  "user_login" smallint NOT NULL,
  "examcredits" integer NOT NULL DEFAULT 7
);

-- --------------------------------------------------------
-- Table structure for table "longqa"
-- --------------------------------------------------------

CREATE TABLE "longqa" (
  "longqa_qid" BIGSERIAL PRIMARY KEY,
  "test_id" varchar(100) NOT NULL,
  "qid" varchar(25) NOT NULL,
  "q" text NOT NULL,
  "marks" integer DEFAULT NULL,
  "uid" bigint DEFAULT NULL,
  CONSTRAINT "longqa_ibfk_1" FOREIGN KEY ("uid") REFERENCES "users" ("uid") ON DELETE SET NULL
);

CREATE INDEX "idx_longqa_uid" ON "longqa" ("uid");

-- --------------------------------------------------------
-- Table structure for table "longtest"
-- --------------------------------------------------------

CREATE TABLE "longtest" (
  "longtest_qid" BIGSERIAL PRIMARY KEY,
  "email" varchar(100) NOT NULL,
  "test_id" varchar(100) NOT NULL,
  "qid" integer NOT NULL,
  "ans" text NOT NULL,
  "marks" integer NOT NULL,
  "uid" bigint NOT NULL,
  CONSTRAINT "longtest_ibfk_1" FOREIGN KEY ("uid") REFERENCES "users" ("uid")
);

CREATE INDEX "idx_longtest_uid" ON "longtest" ("uid");

-- --------------------------------------------------------
-- Table structure for table "practicalqa"
-- --------------------------------------------------------

CREATE TABLE "practicalqa" (
  "pracqa_qid" BIGSERIAL PRIMARY KEY,
  "test_id" varchar(100) NOT NULL,
  "qid" varchar(25) NOT NULL,
  "q" text NOT NULL,
  "compiler" smallint NOT NULL,
  "marks" integer NOT NULL,
  "uid" bigint NOT NULL,
  CONSTRAINT "practicalqa_ibfk_1" FOREIGN KEY ("uid") REFERENCES "users" ("uid")
);

CREATE INDEX "idx_practicalqa_uid" ON "practicalqa" ("uid");

-- --------------------------------------------------------
-- Table structure for table "practicaltest"
-- --------------------------------------------------------

CREATE TABLE "practicaltest" (
  "pid" BIGSERIAL PRIMARY KEY,
  "email" varchar(100) NOT NULL,
  "test_id" varchar(100) NOT NULL,
  "qid" varchar(25) NOT NULL,
  "code" text,
  "input" text,
  "executed" varchar(125) DEFAULT NULL,
  "marks" integer NOT NULL,
  "uid" bigint NOT NULL,
  CONSTRAINT "practicaltest_ibfk_1" FOREIGN KEY ("uid") REFERENCES "users" ("uid")
);

CREATE INDEX "idx_practicaltest_uid" ON "practicaltest" ("uid");

-- --------------------------------------------------------
-- Table structure for table "proctoring_log"
-- --------------------------------------------------------

CREATE TABLE "proctoring_log" (
  "pid" BIGSERIAL PRIMARY KEY,
  "email" varchar(100) NOT NULL,
  "name" varchar(100) NOT NULL,
  "test_id" varchar(100) NOT NULL,
  "voice_db" integer DEFAULT 0,
  "img_log" text NOT NULL,
  "user_movements_updown" smallint NOT NULL,
  "user_movements_lr" smallint NOT NULL,
  "user_movements_eyes" smallint NOT NULL,
  "phone_detection" smallint NOT NULL,
  "person_status" smallint NOT NULL,
  "log_time" timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "uid" bigint NOT NULL,
  CONSTRAINT "proctoring_log_ibfk_1" FOREIGN KEY ("uid") REFERENCES "users" ("uid")
);

CREATE INDEX "idx_proctoring_log_email" ON "proctoring_log" ("email");
CREATE INDEX "idx_proctoring_log_email_test_id" ON "proctoring_log" ("email", "test_id");
CREATE INDEX "idx_proctoring_log_uid" ON "proctoring_log" ("uid");

-- --------------------------------------------------------
-- Table structure for table "questions"
-- --------------------------------------------------------

CREATE TABLE "questions" (
  "questions_uid" BIGSERIAL PRIMARY KEY,
  "test_id" varchar(100) NOT NULL,
  "qid" varchar(25) NOT NULL,
  "q" text NOT NULL,
  "a" varchar(100) NOT NULL,
  "b" varchar(100) NOT NULL,
  "c" varchar(100) NOT NULL,
  "d" varchar(100) NOT NULL,
  "ans" varchar(10) NOT NULL,
  "marks" integer NOT NULL,
  "uid" bigint NOT NULL,
  CONSTRAINT "questions_ibfk_1" FOREIGN KEY ("uid") REFERENCES "users" ("uid")
);

CREATE INDEX "idx_questions_uid" ON "questions" ("uid");

-- --------------------------------------------------------
-- Table structure for table "students"
-- --------------------------------------------------------

CREATE TABLE "students" (
  "sid" BIGSERIAL PRIMARY KEY,
  "email" varchar(100) NOT NULL,
  "test_id" varchar(100) NOT NULL,
  "qid" varchar(25) DEFAULT NULL,
  "ans" text,
  "uid" bigint NOT NULL,
  CONSTRAINT "students_ibfk_1" FOREIGN KEY ("uid") REFERENCES "users" ("uid")
);

CREATE INDEX "idx_students_uid" ON "students" ("uid");

-- --------------------------------------------------------
-- Table structure for table "studenttestinfo"
-- --------------------------------------------------------

CREATE TABLE "studenttestinfo" (
  "stiid" BIGSERIAL PRIMARY KEY,
  "email" varchar(100) NOT NULL,
  "test_id" varchar(100) NOT NULL,
  "time_left" time NOT NULL,
  "completed" smallint DEFAULT 0,
  "uid" bigint NOT NULL,
  CONSTRAINT "studenttestinfo_ibfk_1" FOREIGN KEY ("uid") REFERENCES "users" ("uid")
);

CREATE INDEX "idx_studenttestinfo_uid" ON "studenttestinfo" ("uid");

-- --------------------------------------------------------
-- Table structure for table "teachers"
-- --------------------------------------------------------

CREATE TABLE "teachers" (
  "tid" BIGSERIAL PRIMARY KEY,
  "email" varchar(100) NOT NULL,
  "test_id" varchar(100) NOT NULL,
  "test_type" varchar(75) NOT NULL,
  "start" timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "end" timestamp NOT NULL DEFAULT '1970-01-01 00:00:00',
  "duration" integer NOT NULL,
  "show_ans" integer NOT NULL,
  "password" varchar(100) NOT NULL,
  "subject" varchar(100) NOT NULL,
  "topic" varchar(100) NOT NULL,
  "neg_marks" integer NOT NULL,
  "calc" smallint NOT NULL,
  "proctoring_type" smallint NOT NULL DEFAULT 0,
  "uid" bigint NOT NULL,
  CONSTRAINT "teachers_ibfk_1" FOREIGN KEY ("uid") REFERENCES "users" ("uid")
);

CREATE INDEX "idx_teachers_uid" ON "teachers" ("uid");

-- --------------------------------------------------------
-- Table structure for table "window_estimation_log"
-- --------------------------------------------------------

CREATE TABLE "window_estimation_log" (
  "wid" BIGSERIAL PRIMARY KEY,
  "email" varchar(100) NOT NULL,
  "test_id" varchar(100) NOT NULL,
  "name" varchar(100) NOT NULL,
  "window_event" smallint NOT NULL,
  "transaction_log" timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "uid" bigint NOT NULL,
  CONSTRAINT "window_estimation_log_ibfk_1" FOREIGN KEY ("uid") REFERENCES "users" ("uid")
);

CREATE INDEX "idx_window_estimation_log_uid" ON "window_estimation_log" ("uid");

COMMIT;