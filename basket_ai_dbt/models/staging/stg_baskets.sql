with source as (
  select *
  from {{ source('raw', 'baskets') }}
),

renamed as (
  select
    cast(basket_id as string) as basket_id,

    -- customer_id nullable geliyor; kaybetmemek için sentinel + flag
    coalesce(cast(customer_id as string), '__MISSING__') as customer_id,
    customer_id is null as is_customer_id_missing,

    -- basket_date: INTEGER epoch nanoseconds (ns)
    timestamp_micros(cast(div(basket_date, 1000) as int64)) as basket_ts,
    date(timestamp_micros(cast(div(basket_date, 1000) as int64))) as basket_date,

    cast(total_items as float64) as total_items,
    cast(distinct_items as int64) as distinct_items,
    cast(total_amount as float64) as total_amount,
    cast(category_count as int64) as category_count,

    cast(city as string) as city,
    cast(region as string) as region,
    cast(gender as string) as gender
  from source
)

select *
from renamed
