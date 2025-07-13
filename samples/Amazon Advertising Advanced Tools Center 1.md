---
title: "Amazon Advertising Advanced Tools Center"
source: "https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets"
author:
published:
created: 2025-07-13
description:
tags:
  - "clippings"
---
In this document

- [Targeting types](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#targeting-types)
- [Schema](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#schema)
- [Ad product mapping](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#ad-product-mapping)
- [Representations](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#representations)
- [JSON examples](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#json-examples)
- [Generic](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#generic)
- [Sponsored Products](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#sponsored-products)
- [Sponsored Brands](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#sponsored-brands)
- [Sponsored Display](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#sponsored-display)
- [Auto target](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#auto-target)
- [Keyword target](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#keyword-target)
- [Positive keyword target](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#positive-keyword-target)
- [Negative keyword target (ad group level)](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#negative-keyword-target-ad-group-level)
- [Negative keyword target (campaign level)](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#negative-keyword-target-campaign-level)
- [Product category target](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#product-category-target)
- [Positive product category target](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#positive-product-category-target)
- [Negative product category target (ad group level)](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#negative-product-category-target-ad-group-level)
- [Negative product category target (campaign level)](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#negative-product-category-target-campaign-level)
- [Product target](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#product-target)
- [Positive product target](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#positive-product-target)
- [Negative product target (ad group level)](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#negative-product-target-ad-group-level)
- [Negative product target (campaign level)](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#negative-product-target-campaign-level)
- [Product category audience target](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#product-category-audience-target)
- [Product audience target](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#product-audience-target)
- [Audience target](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#audience-target)

## Target

**Targets** allow advertisers to influence when ads will or will not appear, as well as specify how much to bid for an ad when these conditions are met. Targeting conditions can can be “inclusions” (`negative` = false) or “exclusions” (`negative` = true, also known as negative targets).

## Targeting types

Amazon Ads supports a variety of targeting types depending on the ad product. Some targeting types also have the option to support negative targeting (exclusions). Targets are typically applied at the ad group level, however some types of negative targets are also available at the campaign level.

| Targeting type | Supports negative | Supports campaign-level | Description | Sponsored Products | Sponsored Brands | Sponsored Display |
| --- | --- | --- | --- | --- | --- | --- |
| AUTO | No | No | Targets chosen by Amazon based on a high-level match type or criteria. | x | x | x |
| AUDIENCE | No | No | Targets based on a selected chosen Amazon audience. |  |  | x |
| KEYWORD | Yes | Yes; negative Sponsored Products targets only | Targets based on customer search terms. | x | x |  |
| PRODUCT\_CATEGORY | Yes | Yes; negative Sponsored Products targets only | Targets based on a specified Amazon product category with optional refinements like brand, price, and rating. | x | x | x |
| PRODUCT | Yes | Yes; negative Sponsored Products targets only | Targets based on a specified Amazon product ASIN. | x | x | x |
| PRODUCT\_CATEGORY   \_AUDIENCE | No | No | Targets based on customer views or purchases on an Amazon product category (with optional refinements), within a specified lookback window. |  |  | x |
| PRODUCT\_AUDIENCE | No | No | Targets based on customer views or purchases on an Amazon product ASIN, within a specified lookback window. |  |  | x |
| THEME | No | No | Targets keywords related to a theme. | x |  |  |

## Schema

Targets contains the following fields. **Read-only** indicates that the field is part of the model, but cannot be modified by advertisers. **Required** indicates that a field will always appear in the model.

Note

Some fields are only available for certain ad products. For details, see the [Ad product mapping table](https://advertising.amazon.com/API/docs/en-us/reference/common-models/targets#ad-product-mapping).

| Common field | Type | Required | Read only | Description |
| --- | --- | --- | --- | --- |
| targetId | string | Required | Read Only | The unique identifier of the target. |
| adGroupId | string |  |  | The unique identifier of the ad group that the target belongs to. |
| campaignId | string |  |  | The campaignId associated to the target (for campaign-level targets only). |
| adProduct | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#adproduct) | Required |  | The ad product that the target belongs to. |
| state | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#state) | Required |  | The user set state of the target. |
| negative | boolean | Required |  | Whether to target (false) or exclude (true) the given target. |
| deliveryStatus | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#deliverystatus) | Required | Read Only | This is an overall status if the target is delivering or not. |
| deliveryReasons | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#deliveryreasons) |  | Read Only | This is a list of reasons why the target is not delivering and the reasons behind the delivery status. |
| creationDateTime | datetime | Required | Read Only | The date time that the target was created. |
| lastUpdatedDateTime | datetime | Required | Read Only | The date time that the target was last updated. |
| bid.bid | double |  |  | The bid applied to the target. |
| bid.   currencyCode | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#currencyCode) |  |  | The currency code of the bid applied to the target. |
| targetType | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#targetType) | Required |  | The type of targe. |
| targetDetails.   matchType | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#matchType) |  |  | The match type associated with the target. Differs based on the targetType. |
| targetDetails.   keyword | string |  |  | The keyword text to target. |
| targetDetails.   nativeLanguageKeyword | string |  |  | The unlocalized keyword text in the preferred locale of the advertiser. |
| targetDetails.   nativeLanguageLocale | string |  |  | The locale preference of the advertiser. For example, if the advertiser’s preferred language is Simplified Chinese, set the locale to zh *CN. Supported locales include: Simplified Chinese (locale: zh\_CN) for US, UK and CA. English (locale: en\\* GB) for DE, FR, IT and ES. |
| targetDetails.   productCategoryId | string |  |  | The product category to target. |
| targetDetails.   productCategoryResolved | string |  | Read Only | The resolved human readable name of the category. |
| targetDetails.   productBrand | string |  |  | Refinement to target a specific brand within the product category. |
| targetDetails.   productBrandResolved | string |  | Read Only | The resolved human readable name of the brand. |
| targetDetails.   productGenre | string |  |  | Refinement to target a specific product genre within the product category. |
| targetDetails.   productPriceLessThan | string |  |  | Refinement to target products with a price less than the value within the product category. |
| targetDetails.   productPriceGreaterThan | string |  |  | Refinement to target products with a price greater than the value within the product category. |
| targetDetails.   productRatingLessThan | string |  |  | Refinement to target products with a rating less than the value within the product category. |
| targetDetails.   productRatingGreaterThan | string |  |  | Refinement to target products with a rating greater than the value within the product category. |
| targetDetails.   productAgeRange | string |  |  | Refinement to target products for a specific age range (given as an ID) within the product category. |
| targetDetails.   productAgeRangeResolved | string |  |  | The resolved product age range to target. |
| targetDetails.   productPrimeShippingEligible | boolean |  |  | Refinement to target products that are prime shipping eligible within the product category. |
| targetDetails.   asin | string |  |  | The product asin to target. |
| targetDetails.   event | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#event) |  |  | The product based event to target the audience. |
| targetDetails.   lookback | integer |  |  | The lookback period in days to target the audience for the specified product event. |
| targetDetails.   audienceId | string |  |  | An audience identifier retrieved from the audiences/list resource. |

The mapping table shows how current versions of different ad products map to the common target model. Over time, we will move to standardize the fields in each individual API to the common target model.

| Ad product | Latest version |
| --- | --- |
| Sponsored Products | [Version 3](https://advertising.amazon.com/API/docs/en-us/sponsored-products/3-0) |
| Sponsored Brands | [Version 3](https://advertising.amazon.com/API/docs/en-us/sponsored-brands/3-0) |
| Sponsored Display | [Version 1](https://advertising.amazon.com/API/docs/en-us/sponsored-display/3-0) |

For Amazon Marketing Stream, the table below references the [targets](https://advertising.amazon.com/API/docs/en-us/guides/amazon-marketing-stream/datasets/sponsored-ads-campaign-management#tag/Reports/operation/downloadReport) dataset.

**Legend**

**x**: The ad product uses the common field name in its current version.

**N/A**: The ad product contains the field in the common model schema, but not in its current version schema.

**Empty cell**: The field is not represented for the ad product.

| Common field | Sponsored Products | Sponsored Brands | Sponsored Display | Amazon Marketing Stream |
| --- | --- | --- | --- | --- |
| targetId | keywordId, targetId, negativeKeywordId | keywordId, targetId | x | x |
| adGroupId | x | x | x | x |
| campaignId | x |  |  | x |
| adProduct | N/A | N/A | N/A | x |
| state | x | x | x | x |
| negative | N/A | N/A | N/A | x |
| deliveryStatus | N/A | N/A | N/A |  |
| deliveryReasons | extendedData.   servingStatus | N/A | N/A |  |
| creationDateTime | extendedData.   creationDateTime | N/A | creationDate | audit.   creationDateTime |
| lastUpdatedDateTime | extendedData.   lastUpdateDateTime | N/A | lastUpdateDate | audit.   lastUpdatedDateTime |
| bid.   bid | bid | bid | bid | bid |
| bid.   currencyCode | N/A | N/A | N/A | currencyCode |
| targetType | N/A | N/A | N/A | x |
| targetDetails.   matchType | expression.type (target), matchType (keyword) | expression.type (target), matchType (keyword), themes.   themeType (theme) | expression.type | {targetType}.   matchType |
| targetDetails.   keyword | keywordText | keywordText |  | keywordTarget.   keyword |
| targetDetails.   nativeLanguageKeyword | nativeLanguageKeyword | nativeLanguageKeyword |  | keywordTarget.   nativeLanguageKeyword |
| targetDetails.   nativeLanguageLocale | nativeLanguageLocale | N/A |  | keywordTarget.   nativeLanguageLocale |
| targetDetails.   productCategoryId | expression.value where "type": "ASIN\_CATEGORY\_SAME\_AS" | expressions.   value where "type": "asinCategorySameAs" | expression.value where "type": "asinCategorySameAs" | {targetType}.   targetingClause |
| targetDetails.   productCategoryResolved | resolvedExpression.value where "type": "ASIN\_CATEGORY\_SAME\_AS" | resolvedExpressions.   value where "type": "asinCategorySameAs" | resolvedExpression.value where "type": "asinCategorySameAs" | {targetType}.   targetingClause |
| targetDetails.   productBrand | expression.value where "type": "ASIN\_BRAND\_SAME\_AS" | expressions.   value where "type":"asinBrandSameAs" | expression.value where "type":"asinBrandSameAs" | {targetType}.   targetingClause |
| targetDetails.   productBrandResolved | resolvedExpression.value where "type": "ASIN\_BRAND\_SAME\_AS" | resolvedExpressions.   value where "type":"asinBrandSameAs" | resolvedExpression.value where "type":"asinBrandSameAs" | {targetType}.   targetingClause |
| targetDetails.   productGenre | expression.value where "type": "ASIN\_GENRE\_SAME\_AS" |  | expression.value where "type":"asinGenreSameAs" | {targetType}.   targetingClause |
| targetDetails.   productGenreResolved | resolvedExpression.value where "type": "ASIN\_GENRE\_SAME\_AS" |  | resolvedExpression.value where "type":"asinGenreSameAs" | {targetType}.   targetingClause |
| targetDetails.   productPriceLessThan | expression.value where "type": "ASIN\_PRICE\_LESS\_THAN" or "type": "ASIN\_PRICE\_BETWEEN" (bottom of range) | expressions.   value where "type":"asinPriceLessThan" or "type": "asinPriceBetween" (bottom of range) | expression.value where "type":"asinPriceLessThan" or "type": "asinPriceBetween" (bottom of range) | {targetType}.   targetingClause |
| targetDetails.   productPriceGreaterThan | expression.value where "type": "ASIN\_PRICE\_GREATER\_THAN" or or "type": "ASIN\_PRICE\_BETWEEN" (top of range) | expressions.   value where "type":"asinPriceGreaterThan" or "type": "asinPriceBetween" (top of range) | expression.value where "type":"asinPriceGreaterThan" or "type": "asinPriceBetween" (top of range) | {targetType}.   targetingClause |
| targetDetails.   productRatingLessThan | expression.value where "type": "ASIN\_REVIEW\_RATING\_LESS\_THAN" or "type": "ASIN\_REVIEW\_RATING\_BETWEEN" (bottom of range) | expressions.   value where "type":"asinReviewRatingLessThan" or "type": "asinReviewRatingBetween" (bottom of range) | expression.value where "type":"asinReviewRatingLessThan" or "type": "asinReviewRatingBetween" (bottom of range) | {targetType}.   targetingClause |
| targetDetails.   productRatingGreaterThan | expression.value where "type": "ASIN\_REVIEW\_RATING\_GREATER\_THAN" or "type": "ASIN\_REVIEW\_RATING\_BETWEEN" (top of range) | expressions.   value where "type":"asinReviewRatingGreaterThan" or "type": "asinReviewRatingBetween" (top of range) | expression.value where "type":"asinReviewRatingGreaterThan" or "type": "asinReviewRatingBetween" (top of range) | {targetType}.   targetingClause |
| targetDetails.   productAgeRange | expression.value where "type": "ASIN\_AGE\_RANGE\_SAME\_AS" |  | expression.value where "type": "asinAgeRangeSameAs" | {targetType}.   targetingClause |
| targetDetails.   productAgeRangeResolved | resolvedExpression.value where "type": "ASIN\_AGE\_RANGE\_SAME\_AS" |  | resolvedExpression.value where "type": "asinAgeRangeSameAs" | {targetType}.   targetingClause |
| targetDetails.   productPrimeShippingEligible | expression.value where "type": "ASIN\_IS\_PRIME\_SHIPPING\_ELIGIBLE" |  | expression.value where "type": "asinIsPrimeShippingEligible" | {targetType}.   targetingClause |
| targetDetails.   asin | expression.value where "type": "ASIN\_EXPANDED\_FROM" or "type:" "ASIN\_SAME\_AS" | expression.value where "type:"asinSameAs" | expression.value where "type:"asinSameAs" | {targetType}.   targetingClause |
| targetDetails.   event |  |  | expression.type | {targetType}.   targetingClause |
| targetDetails.   lookback |  |  | expression.value.   value where "expression.value.   type": "views" or "expression.value.   type": "purchases" | {targetType}.   targetingClause |
| targetDetails.   audienceId |  |  | expression.   value.   type.   value where "expression.   value.   type": "audienceSameAs" | {targetType}.targetingClause |

## Representations

The following table shows the different areas where targets are surfaced in the Amazon Ads API.

| Feature | Operations | User guides |
| --- | --- | --- |
| [sp/targets](https://advertising.amazon.com/API/docs/en-us/sponsored-products/3-0/openapi/prod#tag/TargetingClauses) | POST /sp/targets   POST /sp/targets/list   PUT /sp/targets   POST /sp/targets/delete | [SP targeting overview](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-products/product-targeting/overview) |
| [sp/negativeTargets](https://advertising.amazon.com/API/docs/en-us/sponsored-products/3-0/openapi/prod#tag/NegativeTargetingClauses)   [sp/campaignNegativeTargets](https://advertising.amazon.com/API/docs/en-us/sponsored-products/3-0/openapi/prod#tag/campaignNegativeTargetingClauses) | POST /sp/negativeTargets   POST /sp/negativeTargets/list   PUT /sp/negativeTargets   POST /sp/negativeTargets/delete   POST /sp/campaignNegativeTargets   POST /sp/campaignNegativeTargets/list   PUT /sp/campaignNegativeTargets   POST /sp/campaignNegativeTargets/delete | [SP negative targeting overview](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-products/negative-targeting/product-brand) |
| [sp/keywords](https://advertising.amazon.com/API/docs/en-us/sponsored-products/3-0/openapi/prod#tag/Keywords) | POST /sp/keywords   POST /sp/keywords/list   PUT /sp/keywords   POST /sp/keywords/delete | [SP campaign model diagram](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-products/get-started/campaign-structure)   [SP keywords overview](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-products/keywords/overview) |
| [sp/negativeKeywords](https://advertising.amazon.com/API/docs/en-us/sponsored-products/3-0/openapi/prod#tag/NegativeKeywords)   [sp/campaignNegativeKeywords](https://advertising.amazon.com/API/docs/en-us/sponsored-products/3-0/openapi/prod#tag/NegativeKeywords) | POST /sp/negativeKeywords   POST /sp/negativeKeywords/list   PUT /sp/negativeKeywords   POST /sp/negativeKeywords/delete   POST /sp/campaignNegativeKeywords   POST /sp/campaignNegativeKeywords/list   PUT /sp/campaignNegativeKeywords   POST /sp/campaignNegativeKeywords/delete | [SP negative keywords overview](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-products/negative-targeting/keywords) |
| [sb/targets](https://advertising.amazon.com/API/docs/en-us/sponsored-brands/3-0/openapi#tag/Product-targeting) | POST /sb/targets   POST /sb/targets/list   PUT /sb/targets   DELETE /sb/targets/{targetId} | [SB targets overview](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-brands/targeting/product-targeting) |
| [sb/negativeTargets](https://advertising.amazon.com/API/docs/en-us/sponsored-brands/3-0/openapi#tag/Negative-product-targeting) | POST /sb/negativeTargets   POST /sb/negativeTargets/list   PUT /sb/negativeTargets   DELETE /sb/negativeTargets/{targetId} | [SB negative targets overview](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-brands/targeting/negative-product-targeting) |
| [sb/keywords](https://advertising.amazon.com/API/docs/en-us/sponsored-brands/3-0/openapi#tag/Keywords) | POST /sb/keywords   GET /sb/keywords   PUT /sb/keywords   DELETE /sb/keywords/{keywordId} | [SB keywords overview](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-brands/targeting/keyword-targeting) |
| [sb/negativeKeywords](https://advertising.amazon.com/API/docs/en-us/sponsored-brands/3-0/openapi#tag/Negative-keywords) | POST /sb/negativeKeywords   GET /sb/negativeKeywords   PUT /sb/negativeKeywords   DELETE /sb/negativeKeywords/{keywordId} | [SB negative keywords overview](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-brands/targeting/negative-keyword-targeting) |
| [sb/themes](https://advertising.amazon.com/API/docs/en-us/sponsored-brands/3-0/openapi#tag/Theme-targeting) | POST /sb/themes/list   PUT /sb/themes   POST /sb/themes | [Theme targeting overview](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-brands/theme-based-targeting) |
| [sd/targeting](https://advertising.amazon.com/API/docs/en-us/sponsored-display/3-0/openapi#tag/Targeting/operation/listTargetingClauses) | POST /sd/targets   GET sd/targets   GET sd/targets/extended   PUT /sd/targets   DELETE /sd/targets/{targetId} | [SD audiences overview](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-display/audience-targeting)   [SD contextual targeting overview](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-display/contextual-targeting) |
| [sd/negativeTargeting](https://advertising.amazon.com/API/docs/en-us/sponsored-display/3-0/openapi#tag/Negative-Targeting) | POST /sd/negativeTargets   GET sd/negativeTargets   GET sd/negativeTargets/extended   PUT /sd/negativeTargets   DELETE /sd/negativeTargets/{targetId} |  |
| [Amazon Marketing Stream](https://advertising.amazon.com/API/docs/en-us/guides/amazon-marketing-stream/overview) | N/A | [Targets dataset](https://advertising.amazon.com/API/docs/en-us/guides/amazon-marketing-stream/datasets/sponsored-ads-campaign-management#targets-dataset-beta) |
| [targets/exports](https://advertising.amazon.com/API/docs/en-us/exports) | POST /targets/exports | [Exports overview](https://advertising.amazon.com/API/docs/en-us/guides/exports/overview) |

## JSON examples

Below you can find examples of how each ad product represents a target.

### Generic

The generic sample includes a JSON representation of all possible fields in the common schema.

This example includes all possible fields that could be present in a Sponsored Products target, regardless of targetType.

This example includes all possible fields that could be present in a Sponsored Brands target, regardless of targetType.

This example includes all possible fields that could be present in a Sponsored Display target, regardless of targetType.

### Auto target

### Keyword target

#### Positive keyword target

#### Negative keyword target (ad group level)

#### Negative keyword target (campaign level)

### Product category target

#### Positive product category target

#### Negative product category target (ad group level)

#### Negative product category target (campaign level)

### Product target

#### Positive product target

#### Negative product target (ad group level)

#### Negative product target (campaign level)

### Product category audience target

### Product audience target

### Audience target

Was this page helpful?