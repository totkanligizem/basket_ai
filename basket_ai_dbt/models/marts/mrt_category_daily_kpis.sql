with bi as (
  select *
  from {{ ref('stg_basket_items') }}
),

daily as (
  select
    basket_date,
    coalesce(nullif(trim(category_name1), ''), '__UNKNOWN__') as category_name1,

    count(*) as item_rows,
    count(distinct basket_id) as basket_cnt,

    sum(coalesce(amount, 0)) as total_qty,
    sum(coalesce(item_total, 0)) as revenue,

    sum(cast(is_item_code_missing as int64)) as missing_item_code_rows,
    safe_divide(sum(cast(is_item_code_missing as int64)), count(*)) as missing_item_code_rate

  from bi
  group by 1, 2
)

select *
from daily
