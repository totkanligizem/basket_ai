with b as (
  select *
  from {{ ref('stg_baskets') }}
),

bi as (
  select *
  from {{ ref('stg_basket_items') }}
),

agg_items as (
  select
    basket_id,

    -- item kalem sayısı (satır sayısı)
    count(*) as item_rows,

    -- missing item_code kaç satır?
    sum(cast(is_item_code_missing as int64)) as missing_item_code_rows,

    -- distinct ürün sayısı (missing sentinel dahil olmasın diye null'a çeviriyoruz)
    count(distinct nullif(item_code, '__MISSING__')) as distinct_item_codes,

    -- miktar/tutar özetleri
    sum(coalesce(amount, 0)) as total_amount_qty,
    sum(coalesce(item_total, 0)) as total_item_amount,

    -- kategori çeşitliliği (opsiyonel ama faydalı)
    count(distinct nullif(category_name1, '')) as distinct_cat1,
    count(distinct nullif(category_name2, '')) as distinct_cat2,
    count(distinct nullif(category_name3, '')) as distinct_cat3

  from bi
  group by 1
)

select
  b.basket_id,
  b.customer_id,
  b.is_customer_id_missing,
  b.basket_ts,
  b.basket_date,
  b.city,
  b.region,
  b.gender,

  -- basket tablosundaki metrikler
  b.total_items,
  b.distinct_items,
  b.total_amount,
  b.category_count,

  -- items'dan gelen özetler
  a.item_rows,
  a.missing_item_code_rows,
  a.distinct_item_codes,
  a.total_amount_qty,
  a.total_item_amount,
  a.distinct_cat1,
  a.distinct_cat2,
  a.distinct_cat3

from b
left join agg_items a using (basket_id)
