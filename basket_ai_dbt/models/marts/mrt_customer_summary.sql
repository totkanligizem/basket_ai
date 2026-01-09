with s as (
  select *
  from {{ ref('int_basket_summary') }}
),

customer as (
  select
    customer_id,

    -- activity
    count(*) as basket_cnt,
    count(distinct basket_date) as active_days,

    -- dates
    min(basket_date) as first_basket_date,
    max(basket_date) as last_basket_date,

    -- revenue
    sum(coalesce(total_amount, 0)) as revenue_from_baskets,
    sum(coalesce(total_item_amount, 0)) as revenue_from_items,

    avg(coalesce(total_amount, 0)) as aov_from_baskets,
    avg(coalesce(total_item_amount, 0)) as aov_from_items,

    -- items
    sum(coalesce(item_rows, 0)) as item_rows,
    sum(coalesce(distinct_item_codes, 0)) as sum_distinct_items_per_basket,
    avg(coalesce(distinct_item_codes, 0)) as avg_distinct_items_per_basket,

    -- quality flags
    sum(cast(is_customer_id_missing as int64)) as missing_customer_baskets,
    safe_divide(sum(cast(is_customer_id_missing as int64)), count(*)) as missing_customer_rate,

    sum(coalesce(missing_item_code_rows, 0)) as missing_item_code_rows,
    safe_divide(sum(coalesce(missing_item_code_rows, 0)), sum(coalesce(item_rows, 0))) as missing_item_code_rate

  from s
  where customer_id is not null
    and customer_id != '__MISSING__'
  group by 1
)

select *
from customer
