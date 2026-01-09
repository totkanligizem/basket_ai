with source as (
  select *
  from {{ source('raw', 'basket_items') }}
),

renamed as (
  select
    cast(basket_id as string) as basket_id,

    -- customer_id nullable; sentinel + flag
    coalesce(cast(customer_id as string), '__MISSING__') as customer_id,
    customer_id is null as is_customer_id_missing,

    -- basket_date: INTEGER epoch nanoseconds (ns) -> timestamp/date
    timestamp_micros(cast(div(basket_date, 1000) as int64)) as basket_ts,
    date(timestamp_micros(cast(div(basket_date, 1000) as int64))) as basket_date,

    -- itemcode nullable; sentinel + flag
    coalesce(cast(itemcode as string), '__MISSING__') as item_code,
    itemcode is null as is_item_code_missing,

    -- measures
    cast(amount as float64) as amount,
    cast(price as float64) as price,
    cast(item_total as float64) as item_total,

    -- categories
    cast(category_name1 as string) as category_name1,
    cast(category_name2 as string) as category_name2,
    cast(category_name3 as string) as category_name3,

    -- geo/demographics
    cast(city as string) as city,
    cast(region as string) as region,
    cast(gender as string) as gender
  from source
)

select *
from renamed
