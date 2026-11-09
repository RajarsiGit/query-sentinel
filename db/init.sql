-- Schema for the synthetic "shop" workload. Runs once on first container start.
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

CREATE TABLE customers (
    id          bigserial PRIMARY KEY,
    region      text        NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE products (
    id        bigserial PRIMARY KEY,
    category  text           NOT NULL,
    price     numeric(10, 2) NOT NULL
);

CREATE TABLE orders (
    id           bigserial PRIMARY KEY,
    customer_id  bigint         NOT NULL REFERENCES customers (id),
    status       text           NOT NULL,
    total        numeric(12, 2) NOT NULL,
    created_at   timestamptz    NOT NULL DEFAULT now()
);

CREATE TABLE order_items (
    order_id    bigint         NOT NULL REFERENCES orders (id),
    product_id  bigint         NOT NULL REFERENCES products (id),
    qty         int            NOT NULL,
    price       numeric(10, 2) NOT NULL
);

CREATE TABLE events (
    id           bigserial PRIMARY KEY,
    customer_id  bigint      NOT NULL,
    kind         text        NOT NULL,
    payload      jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- Indexes the app "has". Some useful ones (orders.status, events.kind) are
-- deliberately missing so that table growth eventually hurts those queries.
CREATE INDEX orders_customer_idx     ON orders (customer_id, created_at DESC);
CREATE INDEX orders_created_idx      ON orders (created_at);
CREATE INDEX order_items_order_idx   ON order_items (order_id);
CREATE INDEX order_items_product_idx ON order_items (product_id);
CREATE INDEX events_customer_idx     ON events (customer_id, created_at DESC);
CREATE INDEX products_category_idx   ON products (category, price);

-- Where the scorer writes predictions; Grafana reads from here.
CREATE SCHEMA sentinel;
CREATE TABLE sentinel.scores (
    scored_at     timestamptz      NOT NULL DEFAULT now(),
    queryid       bigint           NOT NULL,
    query         text             NOT NULL,
    probability   double precision NOT NULL,
    window_mean_ms double precision NOT NULL,
    baseline_ms   double precision NOT NULL,
    model_version text             NOT NULL,
    PRIMARY KEY (scored_at, queryid)
);
