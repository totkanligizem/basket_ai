# basket_ai_dbt

`basket_ai_dbt` is the analytics engineering layer for the Basket AI project.

## Model Layers

- `staging` (`stg_baskets`, `stg_basket_items`)
- `intermediate` (`int_basket_summary`)
- `marts` (`mrt_daily_kpis`, `mrt_customer_summary`, `mrt_top_items_daily`, `mrt_category_daily_kpis`)

## Notes

- Staging models implement defensive parsing for `basket_date` and support:
  - epoch nanoseconds
  - epoch milliseconds / seconds
  - timestamp/date-compatible raw values
- Missing IDs are preserved with sentinel values and explicit flags.

## Typical Commands

```bash
dbt deps
dbt run
dbt test
```
