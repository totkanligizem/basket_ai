with s as (
  select *
  from {{ ref('int_basket_summary') }}
),

daily as (
  select
    basket_date,

    count(*) as basket_cnt,
    count(distinct customer_id) as customer_cnt,

    -- ciro metrikleri (iki kaynak: basket total_amount ve items total_item_amount)
    sum(coalesce(total_amount, 0)) as revenue_from_baskets,
    sum(coalesce(total_item_amount, 0)) as revenue_from_items,

    avg(coalesce(total_amount, 0)) as aov_from_baskets,
    avg(coalesce(total_item_amount, 0)) as aov_from_items,

    avg(coalesce(distinct_item_codes, 0)) as avg_distinct_items,

    -- missing kalite metrikleri
    sum(coalesce(missing_item_code_rows, 0)) as missing_item_code_rows,
    sum(coalesce(item_rows, 0)) as item_rows,
    safe_divide(sum(coalesce(missing_item_code_rows, 0)), sum(coalesce(item_rows, 0))) as missing_item_code_rate,

    sum(cast(is_customer_id_missing as int64)) as missing_customer_baskets,
    safe_divide(sum(cast(is_customer_id_missing as int64)), count(*)) as missing_customer_rate

  from s
  group by 1
)

select *
from daily
order by basket_date
