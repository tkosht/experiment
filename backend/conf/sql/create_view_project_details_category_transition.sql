create or replace view project_details_category_transition
as
select sortby, created_at,split_part(data_category, '_', 2) as data_category, count(*) as cnt
from project_details pd
group by sortby, data_category, created_at
order by created_at desc, sortby , cnt desc

