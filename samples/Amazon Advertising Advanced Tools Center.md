---
title: "Amazon Advertising Advanced Tools Center"
source: "https://advertising.amazon.com/API/docs/en-us/reference/common-models/ad-groups"
author:
published:
created: 2025-07-13
description:
tags:
  - "clippings"
---
**Ad groups** are used to group ads that have common targeting, strategy, and creatives.

## Schema

Ad groups contain the following fields. **Read-only** indicates that the field is part of the model, but cannot be modified by advertisers. **Required** indicates that a field will always appear as part of model.

Note

Some ad group fields are only available for certain ad products. For details, see the [Ad product mapping table](https://advertising.amazon.com/API/docs/en-us/reference/common-models/ad-groups#ad-product-mapping).

| Field | Description | Type | Required | Read only |
| --- | --- | --- | --- | --- |
| adGroupId | The unique identifier of the ad group. | String | TRUE | TRUE |
| campaignId | The unique identifier of the campaign the ad group belongs to. | String | TRUE |  |
| adProduct | The ad product that the ad group belongs to. | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#adproduct) | TRUE |  |
| name | The name of the ad group. | String | TRUE |  |
| state | The user-set state of the ad group. | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#state) | TRUE |  |
| deliveryStatus | This is an overall status if the ad group is delivering or not. | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#deliverystatus) | TRUE | TRUE |
| deliveryReasons | This is a list of reasons why the ad group is not delivering and the reasons behind the delivery status. | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#deliveryreasons) |  | TRUE |
| creativeType | The creative type that this ad group contains. | String |  |  |
| creationDateTime | The date time that the ad group was created. | datetime | TRUE | TRUE |
| lastUpdatedDateTime | The date time that the ad group was last updated. | datetime | TRUE | TRUE |
| bid.   defaultBid | The default maximum bid for ads and targets in the ad group. This is used in sponsored ads as the maximum bid during the auction. | double | TRUE |  |
| bid.   currencyCode | The currency code the bid is expressed in. | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#currencyCode) | TRUE |  |
| optimization.   goalSetting.   goal | The type of goal associated with the ad group. | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#goal) |  |  |
| optimization.   goalSetting.   kpi | The way the goal associated with the ad group is measured. | [Enum](https://advertising.amazon.com/API/docs/en-us/reference/common-models/enums#kpi) |  |  |

The mapping table shows how current versions of different ad products map to the common ad group model. Over time, we will move to standardize the fields in each individual API to the common ad group model.

| Ad product | Latest version |
| --- | --- |
| Sponsored Products | [Version 3](https://advertising.amazon.com/API/docs/en-us/sponsored-products/3-0/openapi/prod#tag/AdGroups) |
| Sponsored Brands | [Version 4](https://advertising.amazon.com/API/docs/en-us/sponsored-brands/3-0/openapi/prod#tag/AdGroups) |
| Sponsored Display | [Version 1](https://advertising.amazon.com/API/docs/en-us/sponsored-display/3-0/openapi#tag/Ad-Groups) |
| Sponsored Television | Version 1 |

For Amazon Marketing Stream, the table below references the [ad groups](https://advertising.amazon.com/API/docs/en-us/guides/amazon-marketing-stream/datasets/sponsored-ads-campaign-management#ad-groups-dataset-beta) dataset.

**Legend**

**x**: The ad product uses the common field name in its current version.

**N/A**: The ad product contains the field in the common model schema, but not in its current version schema.

**Empty cell**: The field is not represented for the ad product.

| Common field | Sponsored Products | Sponsored Brands | Sponsored Display | Amazon Marketing Stream | Sponsored TV |
| --- | --- | --- | --- | --- | --- |
| adGroupId | x | x | x | x | x |
| campaignId | x | x | x | x | x |
| adProduct | N/A | N/A | N/A | x | N/A |
| name | x | x | x | x | x |
| state | x | x | x | x | x |
| deliveryStatus | N/A | N/A | servingStatus |  | N/A |
| deliveryReasons | extendedData.   servingStatus | extendedData.   servingStatus | N/A |  |  |
| creativeType |  |  | x |  |  |
| creationDateTime | extendedData.creationDateTime | extendedData.   creationDate | creationDate | audit.   creationDateTime | extendedData.   creationDate |
| lastUpdatedDateTime | extendedData.   lastUpdateDateTime | extendedData.   lastUpdateDate | lastUpdateDate | audit.   lastUpdatedDateTime | extendedData.   lastUpdateDate |
| bid.   defaultBid | defaultBid |  | defaultBid | bidValue.   defaultBid.value | defaultBid.bid |
| bid.   currencyCode | N/A |  | N/A | bidValue.   defaultBid.   currencyCode |  |
| optimization.   goalSettings.   goal |  |  | bidOptimization |  |  |
| optimization.   goalSettings.   Kpi |  |  | derived from bidOptimization |  |  |

## Representations

The following table shows the different areas where ad groups are surfaced in the Amazon Ads API.

| Feature | Operations | User guides |
| --- | --- | --- |
| [sp/adGroups](https://advertising.amazon.com/API/docs/en-us/sponsored-products/3-0/openapi/prod#tag/AdGroups) | POST /sp/adGroups   POST /sp/adGroups/list   PUT /sp/adGroups   POST /sp/adGroups/delete | [SP campaign model diagram](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-products/get-started/campaign-structure)   [SP ad groups overview](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-products/ad-groups) |
| [sb/v4/adGroups](https://advertising.amazon.com/API/docs/en-us/sponsored-brands/3-0/openapi/prod#tag/AdGroups) | POST /sb/v4/adGroups   POST /sb/v4/adGroups/list   PUT /sb/v4/adGroups   POST /sb/v4/adGroups/delete | [SB campaign model diagram](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-brands/campaigns/structure) |
| [sd/adGroups](https://advertising.amazon.com/API/docs/en-us/sponsored-display/3-0/openapi#tag/Ad-Groups) | POST /sd/adGroups   GET sd/adGroups   PUT /sd/adGroups   DELETE /sd/adGroups/{adGroupId} | [SD campaigns overview](https://advertising.amazon.com/API/docs/en-us/guides/sponsored-display/overview) |
| [Amazon Marketing Stream](https://advertising.amazon.com/API/docs/en-us/guides/amazon-marketing-stream/overview) | N/A | [Ad groups dataset](https://advertising.amazon.com/API/docs/en-us/guides/amazon-marketing-stream/datasets/ad-groups) |
| [adGroups/exports](https://advertising.amazon.com/API/docs/en-us/exports) | POST /adGroups/exports | [Exports overview](https://advertising.amazon.com/API/docs/en-us/guides/exports/overview) |
| st/adGroups | POST /st/adGroups   POST /st/adGroups/list   PUT /st/adGroups   POST /st/adGroups/delete |  |

## JSON examples

Below you can find examples of how each ad product represents an ad group.

### Generic

The generic sample includes a JSON representation of all possible fields in the common schema.

```json
[
    {
        "adGroupId": "string",
        "campaignId": "string",
        "adProduct": "string",
        "name": "string",
        "state": "string",
        "deliveryStatus": "string",
        "deliveryReasons": list<string>,
        "creativeType": "string",
        "creationDateTime": datetime,
        "lastUpdatedDateTime": datetime,
        "bid": {
            "defaultBid": 0.0,
            "currencyCode": "string"
        },
        "optimization": {
            "goalSettings": {
                "goalType": "string",
                "goalKpi": "string"
            }
        }
    }
]
```

Was this page helpful?