with source as (
  select *
  from {{ source('raw', 'baskets') }}
),

typed as (
  select
    *,
    safe_cast(basket_date as int64) as basket_date_i64,
    safe_cast(basket_date as timestamp) as basket_date_ts_raw
  from source
),

renamed as (
  select
    cast(basket_id as string) as basket_id,

    -- customer_id nullable geliyor; kaybetmemek için sentinel + flag
    coalesce(cast(customer_id as string), '__MISSING__') as customer_id,
    customer_id is null as is_customer_id_missing,

    -- basket_date güvenli parse:
    -- - epoch ns/ms/s integer
    -- - timestamp/date string
    case
      when basket_date_i64 is null then basket_date_ts_raw
      when basket_date_i64 > 2000000000000000000 then timestamp_micros(cast(div(basket_date_i64, 1000) as int64))
      when basket_date_i64 > 2000000000000000 then timestamp_micros(basket_date_i64)
      when basket_date_i64 > 2000000000000 then timestamp_millis(basket_date_i64)
      else timestamp_seconds(basket_date_i64)
    end as basket_ts,
    date(
      case
        when basket_date_i64 is null then basket_date_ts_raw
        when basket_date_i64 > 2000000000000000000 then timestamp_micros(cast(div(basket_date_i64, 1000) as int64))
        when basket_date_i64 > 2000000000000000 then timestamp_micros(basket_date_i64)
        when basket_date_i64 > 2000000000000 then timestamp_millis(basket_date_i64)
        else timestamp_seconds(basket_date_i64)
      end
    ) as basket_date,

    cast(total_items as float64) as total_items,
    cast(distinct_items as int64) as distinct_items,
    cast(total_amount as float64) as total_amount,
    cast(category_count as int64) as category_count,

    cast(city as string) as city,
    cast(region as string) as region,
    cast(gender as string) as gender
  from typed
)

select *
from renamed
