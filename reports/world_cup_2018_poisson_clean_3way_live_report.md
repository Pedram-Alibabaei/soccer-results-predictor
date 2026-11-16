# World Cup 2018 Poisson goal baseline Report (clean_3way, live tournament ratings)

## Data

- Train rows: 41463
- Train date range: 1872-11-30 to 2018-06-12
- Test rows: 64
- Test date range: 2018-06-14 to 2018-07-15

## Headline Metrics

| Model | Accuracy | Log loss | Brier |
| --- | ---: | ---: | ---: |
| Majority baseline | 0.375 | 1.106 | 0.676 |
| Poisson goal baseline | 0.594 | 0.947 | 0.561 |

## Confusion Matrix

Rows are actual results; columns are predicted results.

| actual | 1 | X | 2 |
| --- | --- | --- | --- |
| 1 | 21 | 0 | 3 |
| X | 6 | 0 | 8 |
| 2 | 9 | 0 | 17 |

## Predicted Result Distribution

| result | count |
| --- | --- |
| 1 | 36 |
| X | 0 |
| 2 | 28 |

## Average Predicted Probabilities

| result | probability |
| --- | --- |
| 1 | 0.403 |
| X | 0.251 |
| 2 | 0.346 |

## Highest-Confidence Misses

| date | home_team | away_team | home_goals | away_goals | result | world_cup_stage | predicted | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018-06-27 | South Korea | Germany | 2 | 0 | 1 | group | 2 | 0.685 |
| 2018-06-16 | Argentina | Iceland | 1 | 1 | X | group | 1 | 0.676 |
| 2018-06-25 | Spain | Morocco | 2 | 2 | X | group | 1 | 0.649 |
| 2018-07-06 | Brazil | Belgium | 1 | 2 | 2 | quarter_final | 1 | 0.634 |
| 2018-06-17 | Brazil | Switzerland | 1 | 1 | X | group | 1 | 0.628 |
| 2018-06-19 | Colombia | Japan | 1 | 2 | 2 | group | 1 | 0.610 |
| 2018-07-01 | Russia | Spain | 1 | 1 | X | round_of_16 | 2 | 0.596 |
| 2018-06-27 | Switzerland | Costa Rica | 2 | 2 | X | group | 1 | 0.567 |
| 2018-06-21 | Denmark | Australia | 1 | 1 | X | group | 1 | 0.547 |
| 2018-06-15 | Portugal | Spain | 3 | 3 | X | group | 2 | 0.542 |

## Match Predictions

| date | home_team | away_team | home_goals | away_goals | result | world_cup_stage | predicted | p_1 | p_X | p_2 | home_xg | away_xg | correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018-06-14 | Russia | Saudi Arabia | 5 | 0 | 1 | group | 1 | 0.5641 | 0.2494 | 0.1865 | 1.6030 | 0.8010 | True |
| 2018-06-15 | Egypt | Uruguay | 0 | 1 | 2 | group | 2 | 0.1807 | 0.2769 | 0.5424 | 0.6730 | 1.3900 | True |
| 2018-06-15 | Morocco | Iran | 0 | 1 | 2 | group | 2 | 0.3316 | 0.2875 | 0.3809 | 1.0720 | 1.1710 | True |
| 2018-06-15 | Portugal | Spain | 3 | 3 | X | group | 2 | 0.2171 | 0.2407 | 0.5422 | 0.9770 | 1.6940 | False |
| 2018-06-16 | Argentina | Iceland | 1 | 1 | X | group | 1 | 0.6758 | 0.1943 | 0.1298 | 2.0950 | 0.7900 | False |
| 2018-06-16 | Croatia | Nigeria | 2 | 0 | 1 | group | 1 | 0.5096 | 0.2695 | 0.2209 | 1.4340 | 0.8430 | True |
| 2018-06-16 | France | Australia | 2 | 1 | 1 | group | 1 | 0.6206 | 0.2218 | 0.1576 | 1.8420 | 0.8010 | True |
| 2018-06-16 | Peru | Denmark | 0 | 1 | 2 | group | 1 | 0.4328 | 0.2939 | 0.2734 | 1.2110 | 0.8980 | False |
| 2018-06-17 | Brazil | Switzerland | 1 | 1 | X | group | 1 | 0.6279 | 0.2329 | 0.1393 | 1.7080 | 0.6570 | False |
| 2018-06-17 | Costa Rica | Serbia | 0 | 1 | 2 | group | 1 | 0.3753 | 0.2750 | 0.3497 | 1.2380 | 1.1840 | False |
| 2018-06-17 | Germany | Mexico | 0 | 1 | 2 | group | 1 | 0.4486 | 0.2824 | 0.2690 | 1.2990 | 0.9370 | False |
| 2018-06-18 | Belgium | Panama | 3 | 0 | 1 | group | 1 | 0.6619 | 0.2097 | 0.1283 | 1.9120 | 0.6980 | True |
| 2018-06-18 | Sweden | South Korea | 1 | 0 | 1 | group | 1 | 0.4539 | 0.2948 | 0.2513 | 1.2290 | 0.8340 | True |
| 2018-06-18 | Tunisia | England | 1 | 2 | 2 | group | 2 | 0.1210 | 0.2314 | 0.6475 | 0.5720 | 1.6930 | True |
| 2018-06-19 | Colombia | Japan | 1 | 2 | 2 | group | 1 | 0.6104 | 0.2309 | 0.1587 | 1.7520 | 0.7610 | False |
| 2018-06-19 | Poland | Senegal | 1 | 2 | 2 | group | 1 | 0.3780 | 0.2812 | 0.3409 | 1.2030 | 1.1280 | False |
| 2018-06-19 | Russia | Egypt | 3 | 1 | 1 | group | 1 | 0.4928 | 0.2829 | 0.2243 | 1.3320 | 0.7990 | True |
| 2018-06-20 | Iran | Spain | 0 | 1 | 2 | group | 2 | 0.2056 | 0.2445 | 0.5500 | 0.9060 | 1.6540 | True |
| 2018-06-20 | Portugal | Morocco | 1 | 0 | 1 | group | 1 | 0.5111 | 0.2532 | 0.2358 | 1.5700 | 0.9780 | True |
| 2018-06-20 | Uruguay | Saudi Arabia | 1 | 0 | 1 | group | 1 | 0.7289 | 0.1795 | 0.0916 | 2.1280 | 0.6000 | True |
| 2018-06-21 | Argentina | Croatia | 0 | 3 | 2 | group | 1 | 0.4962 | 0.2597 | 0.2441 | 1.5060 | 0.9720 | False |
| 2018-06-21 | Denmark | Australia | 1 | 1 | X | group | 1 | 0.5469 | 0.2547 | 0.1984 | 1.5600 | 0.8260 | False |
| 2018-06-21 | France | Peru | 1 | 0 | 1 | group | 1 | 0.4273 | 0.2725 | 0.3001 | 1.3400 | 1.0760 | True |
| 2018-06-22 | Brazil | Costa Rica | 2 | 0 | 1 | group | 1 | 0.7817 | 0.1580 | 0.0602 | 2.2130 | 0.4450 | True |
| 2018-06-22 | Nigeria | Iceland | 2 | 0 | 1 | group | 1 | 0.3712 | 0.2679 | 0.3609 | 1.2780 | 1.2560 | True |
| 2018-06-22 | Serbia | Switzerland | 1 | 2 | 2 | group | 2 | 0.2195 | 0.2575 | 0.5229 | 0.8940 | 1.5340 | True |
| 2018-06-23 | Belgium | Tunisia | 5 | 2 | 1 | group | 1 | 0.6599 | 0.2052 | 0.1349 | 1.9780 | 0.7600 | True |
| 2018-06-23 | Germany | Sweden | 2 | 1 | 1 | group | 1 | 0.4921 | 0.2791 | 0.2288 | 1.3570 | 0.8290 | True |
| 2018-06-23 | South Korea | Mexico | 1 | 2 | 2 | group | 2 | 0.1775 | 0.2639 | 0.5586 | 0.7060 | 1.4820 | True |
| 2018-06-24 | England | Panama | 6 | 1 | 1 | group | 1 | 0.7162 | 0.1948 | 0.0890 | 1.9490 | 0.5190 | True |
| 2018-06-24 | Japan | Senegal | 2 | 2 | X | group | 2 | 0.2866 | 0.2696 | 0.4438 | 1.0550 | 1.3820 | False |
| 2018-06-24 | Poland | Colombia | 0 | 3 | 2 | group | 2 | 0.2726 | 0.2642 | 0.4632 | 1.0430 | 1.4450 | True |
| 2018-06-25 | Iran | Portugal | 1 | 1 | X | group | 2 | 0.2956 | 0.2750 | 0.4294 | 1.0500 | 1.3250 | False |
| 2018-06-25 | Russia | Uruguay | 0 | 3 | 2 | group | 2 | 0.2659 | 0.2784 | 0.4556 | 0.9470 | 1.3320 | True |
| 2018-06-25 | Saudi Arabia | Egypt | 2 | 1 | 1 | group | 2 | 0.2641 | 0.2717 | 0.4642 | 0.9750 | 1.3880 | False |
| 2018-06-25 | Spain | Morocco | 2 | 2 | X | group | 1 | 0.6493 | 0.2103 | 0.1404 | 1.9340 | 0.7660 | False |
| 2018-06-26 | Australia | Peru | 0 | 2 | 2 | group | 2 | 0.1961 | 0.2486 | 0.5553 | 0.8460 | 1.6130 | True |
| 2018-06-26 | Denmark | France | 0 | 0 | X | group | 2 | 0.2317 | 0.2722 | 0.4962 | 0.8680 | 1.4080 | False |
| 2018-06-26 | Iceland | Croatia | 1 | 2 | 2 | group | 2 | 0.1788 | 0.2255 | 0.5956 | 0.8900 | 1.8310 | True |
| 2018-06-26 | Nigeria | Argentina | 1 | 2 | 2 | group | 2 | 0.1546 | 0.2317 | 0.6137 | 0.7370 | 1.7390 | True |
| 2018-06-27 | Mexico | Sweden | 0 | 3 | 2 | group | 1 | 0.4264 | 0.2979 | 0.2757 | 1.1810 | 0.8880 | False |
| 2018-06-27 | Serbia | Brazil | 0 | 2 | 2 | group | 2 | 0.0499 | 0.1322 | 0.8179 | 0.4560 | 2.4820 | True |
| 2018-06-27 | South Korea | Germany | 2 | 0 | 1 | group | 2 | 0.1107 | 0.2039 | 0.6855 | 0.6190 | 1.9260 | False |
| 2018-06-27 | Switzerland | Costa Rica | 2 | 2 | X | group | 1 | 0.5670 | 0.2534 | 0.1796 | 1.5660 | 0.7550 | False |
| 2018-06-28 | England | Belgium | 0 | 1 | 2 | group | 1 | 0.4044 | 0.2540 | 0.3416 | 1.4510 | 1.3130 | False |
| 2018-06-28 | Japan | Poland | 0 | 1 | 2 | group | 2 | 0.3146 | 0.2505 | 0.4349 | 1.2650 | 1.5310 | True |
| 2018-06-28 | Panama | Tunisia | 1 | 2 | 2 | group | 2 | 0.3130 | 0.2582 | 0.4289 | 1.2040 | 1.4540 | True |
| 2018-06-28 | Senegal | Colombia | 0 | 1 | 2 | group | 2 | 0.2686 | 0.2715 | 0.4599 | 0.9900 | 1.3850 | True |
| 2018-06-30 | France | Argentina | 4 | 3 | 1 | round_of_16 | 1 | 0.4436 | 0.2708 | 0.2857 | 1.3730 | 1.0450 | True |
| 2018-06-30 | Uruguay | Portugal | 2 | 1 | 1 | round_of_16 | 1 | 0.4063 | 0.2758 | 0.3180 | 1.2870 | 1.1050 | True |
| 2018-07-01 | Croatia | Denmark | 1 | 1 | X | round_of_16 | 1 | 0.3694 | 0.2959 | 0.3347 | 1.1050 | 1.0360 | False |
| 2018-07-01 | Russia | Spain | 1 | 1 | X | round_of_16 | 2 | 0.1731 | 0.2305 | 0.5964 | 0.8330 | 1.7720 | False |
| 2018-07-02 | Belgium | Japan | 3 | 2 | 1 | round_of_16 | 1 | 0.6902 | 0.1888 | 0.1210 | 2.1360 | 0.7630 | True |
| 2018-07-02 | Brazil | Mexico | 2 | 0 | 1 | round_of_16 | 1 | 0.6696 | 0.2206 | 0.1098 | 1.7650 | 0.5510 | True |
| 2018-07-03 | Colombia | England | 1 | 1 | X | round_of_16 | 2 | 0.2970 | 0.2861 | 0.4169 | 0.9960 | 1.2360 | False |
| 2018-07-03 | Sweden | Switzerland | 1 | 0 | 1 | round_of_16 | 2 | 0.3220 | 0.2943 | 0.3837 | 1.0160 | 1.1380 | False |
| 2018-07-06 | Brazil | Belgium | 1 | 2 | 2 | quarter_final | 1 | 0.6339 | 0.2131 | 0.1530 | 1.9300 | 0.8230 | False |
| 2018-07-06 | Uruguay | France | 0 | 2 | 2 | quarter_final | 2 | 0.3222 | 0.2757 | 0.4021 | 1.1150 | 1.2800 | True |
| 2018-07-07 | Russia | Croatia | 1 | 1 | X | quarter_final | 2 | 0.3276 | 0.2826 | 0.3898 | 1.0890 | 1.2160 | False |
| 2018-07-07 | Sweden | England | 0 | 2 | 2 | quarter_final | 2 | 0.2655 | 0.2968 | 0.4377 | 0.8650 | 1.2000 | True |
| 2018-07-10 | France | Belgium | 1 | 0 | 1 | semi_final | 1 | 0.4152 | 0.2511 | 0.3337 | 1.4960 | 1.3150 | True |
| 2018-07-11 | Croatia | England | 1 | 1 | X | semi_final | 2 | 0.2340 | 0.2689 | 0.4971 | 0.8910 | 1.4330 | False |
| 2018-07-14 | Belgium | England | 2 | 0 | 1 | third_place | 1 | 0.3714 | 0.2702 | 0.3584 | 1.2620 | 1.2350 | True |
| 2018-07-15 | France | Croatia | 4 | 2 | 1 | final | 1 | 0.5667 | 0.2393 | 0.1940 | 1.7000 | 0.8840 | True |
