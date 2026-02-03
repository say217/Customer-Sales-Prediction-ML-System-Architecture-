def test_no_missing_columns(sample_df):
    required = {
        "age", "annual_income", "website_visits",
        "time_on_site", "discount_rate", "past_purchases",
        "region", "device_type", "membership_level",
        "monthly_sales"
    }
    assert required.issubset(sample_df.columns)
