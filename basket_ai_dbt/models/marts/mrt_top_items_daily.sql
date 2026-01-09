with bi as (
  select *
  from {{ ref('stg_basket_items') }}
),

daily as (
  select
    basket_date,
    item_code,

    count(*) as item_rows,
    count(distinct basket_id) as basket_cnt,

    sum(coalesce(amount, 0)) as total_qty,
    sum(coalesce(item_total, 0)) as revenue,

    sum(cast(is_item_code_missing as int64)) as missing_item_code_rows

  from bi
  group by 1, 2
),

ranked as (
  select
    *,
    dense_rank() over (
      partition by basket_date
      order by revenue desc, basket_cnt desc, item_rows desc
    ) as revenue_rank
  from daily
)

select
  basket_date,
  item_code,
  revenue_rank,
  basket_cnt,
  item_rows,
  total_qty,
  revenue,
  missing_item_code_rows,
  safe_divide(missing_item_code_rows, item_rows) as missing_item_code_rate
from ranked
where revenue_rank <= 20
