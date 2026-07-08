# World Cup 2018 Poisson goal baseline Report (clean_3way)

## Data

- Train rows: 41463
- Train date range: 1872-11-30 to 2018-06-12
- Test rows: 64
- Test date range: 2018-06-14 to 2018-07-15

## Headline Metrics

| Model | Accuracy | Log loss | Brier |
| --- | ---: | ---: | ---: |
| Majority baseline | 0.375 | 1.106 | 0.676 |
| Poisson goal baseline | 0.562 | 0.951 | 0.563 |

## Confusion Matrix

Rows are actual results; columns are predicted results.

| actual | 1 | X | 2 |
| --- | --- | --- | --- |
| 1 | 19 | 0 | 5 |
| X | 6 | 0 | 8 |
| 2 | 9 | 0 | 17 |

## Predicted Result Distribution

| result | count |
| --- | --- |
| 1 | 34 |
| X | 0 |
| 2 | 30 |

## Average Predicted Probabilities

| result | probability |
| --- | --- |
| 1 | 0.393 |
| X | 0.252 |
| 2 | 0.355 |

## Highest-Confidence Misses

| date | home_team | away_team | home_goals | away_goals | result | world_cup_stage | predicted | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018-06-27 | South Korea | Germany | 2 | 0 | 1 | group | 2 | 0.686 |
| 2018-06-16 | Argentina | Iceland | 1 | 1 | X | group | 1 | 0.676 |
| 2018-07-06 | Brazil | Belgium | 1 | 2 | 2 | quarter_final | 1 | 0.660 |
| 2018-06-25 | Spain | Morocco | 2 | 2 | X | group | 1 | 0.638 |
| 2018-06-17 | Brazil | Switzerland | 1 | 1 | X | group | 1 | 0.628 |
| 2018-07-01 | Russia | Spain | 1 | 1 | X | round_of_16 | 2 | 0.621 |
| 2018-06-19 | Colombia | Japan | 1 | 2 | 2 | group | 1 | 0.610 |
| 2018-06-15 | Portugal | Spain | 3 | 3 | X | group | 2 | 0.542 |
| 2018-06-27 | Switzerland | Costa Rica | 2 | 2 | X | group | 1 | 0.540 |
| 2018-06-21 | Argentina | Croatia | 0 | 3 | 2 | group | 1 | 0.521 |

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
| 2018-06-19 | Russia | Egypt | 3 | 1 | 1 | group | 1 | 0.4578 | 0.2908 | 0.2514 | 1.2550 | 0.8500 | True |
| 2018-06-20 | Iran | Spain | 0 | 1 | 2 | group | 2 | 0.1736 | 0.2329 | 0.5935 | 0.8230 | 1.7480 | True |
| 2018-06-20 | Portugal | Morocco | 1 | 0 | 1 | group | 1 | 0.5020 | 0.2571 | 0.2409 | 1.5310 | 0.9750 | True |
| 2018-06-20 | Uruguay | Saudi Arabia | 1 | 0 | 1 | group | 1 | 0.6788 | 0.2028 | 0.1184 | 1.9600 | 0.6710 | True |
| 2018-06-21 | Argentina | Croatia | 0 | 3 | 2 | group | 1 | 0.5214 | 0.2535 | 0.2250 | 1.5690 | 0.9360 | False |
| 2018-06-21 | Denmark | Australia | 1 | 1 | X | group | 1 | 0.5135 | 0.2618 | 0.2248 | 1.4960 | 0.8930 | False |
| 2018-06-21 | France | Peru | 1 | 0 | 1 | group | 1 | 0.4021 | 0.2708 | 0.3270 | 1.3140 | 1.1570 | True |
| 2018-06-22 | Brazil | Costa Rica | 2 | 0 | 1 | group | 1 | 0.7890 | 0.1526 | 0.0585 | 2.2700 | 0.4510 | True |
| 2018-06-22 | Nigeria | Iceland | 2 | 0 | 1 | group | 1 | 0.4098 | 0.2623 | 0.3279 | 1.3910 | 1.2160 | True |
| 2018-06-22 | Serbia | Switzerland | 1 | 2 | 2 | group | 2 | 0.2081 | 0.2498 | 0.5421 | 0.8890 | 1.6040 | True |
| 2018-06-23 | Belgium | Tunisia | 5 | 2 | 1 | group | 1 | 0.6324 | 0.2151 | 0.1524 | 1.9060 | 0.8090 | True |
| 2018-06-23 | Germany | Sweden | 2 | 1 | 1 | group | 1 | 0.5284 | 0.2665 | 0.2051 | 1.4620 | 0.7990 | True |
| 2018-06-23 | South Korea | Mexico | 1 | 2 | 2 | group | 2 | 0.2006 | 0.2671 | 0.5323 | 0.7800 | 1.4580 | True |
| 2018-06-24 | England | Panama | 6 | 1 | 1 | group | 1 | 0.6932 | 0.2075 | 0.0993 | 1.8580 | 0.5370 | True |
| 2018-06-24 | Japan | Senegal | 2 | 2 | X | group | 2 | 0.2755 | 0.2707 | 0.4538 | 1.0150 | 1.3850 | False |
| 2018-06-24 | Poland | Colombia | 0 | 3 | 2 | group | 2 | 0.2649 | 0.2641 | 0.4710 | 1.0190 | 1.4530 | True |
| 2018-06-25 | Iran | Portugal | 1 | 1 | X | group | 2 | 0.2930 | 0.2715 | 0.4354 | 1.0620 | 1.3580 | False |
| 2018-06-25 | Russia | Uruguay | 0 | 3 | 2 | group | 2 | 0.2447 | 0.2770 | 0.4783 | 0.8890 | 1.3620 | True |
| 2018-06-25 | Saudi Arabia | Egypt | 2 | 1 | 1 | group | 2 | 0.2686 | 0.2757 | 0.4556 | 0.9680 | 1.3510 | False |
| 2018-06-25 | Spain | Morocco | 2 | 2 | X | group | 1 | 0.6380 | 0.2109 | 0.1512 | 1.9510 | 0.8250 | False |
| 2018-06-26 | Australia | Peru | 0 | 2 | 2 | group | 2 | 0.1722 | 0.2320 | 0.5958 | 0.8210 | 1.7560 | True |
| 2018-06-26 | Denmark | France | 0 | 0 | X | group | 2 | 0.2410 | 0.2694 | 0.4897 | 0.9120 | 1.4250 | False |
| 2018-06-26 | Iceland | Croatia | 1 | 2 | 2 | group | 2 | 0.2425 | 0.2444 | 0.5131 | 1.0580 | 1.6550 | True |
| 2018-06-26 | Nigeria | Argentina | 1 | 2 | 2 | group | 2 | 0.1132 | 0.2011 | 0.6857 | 0.6470 | 1.9630 | True |
| 2018-06-27 | Mexico | Sweden | 0 | 3 | 2 | group | 1 | 0.3989 | 0.3009 | 0.3002 | 1.1270 | 0.9360 | False |
| 2018-06-27 | Serbia | Brazil | 0 | 2 | 2 | group | 2 | 0.0460 | 0.1235 | 0.8304 | 0.4570 | 2.5800 | True |
| 2018-06-27 | South Korea | Germany | 2 | 0 | 1 | group | 2 | 0.1131 | 0.2006 | 0.6863 | 0.6480 | 1.9680 | False |
| 2018-06-27 | Switzerland | Costa Rica | 2 | 2 | X | group | 1 | 0.5400 | 0.2585 | 0.2015 | 1.5270 | 0.8200 | False |
| 2018-06-28 | England | Belgium | 0 | 1 | 2 | group | 1 | 0.4070 | 0.2701 | 0.3229 | 1.3270 | 1.1510 | False |
| 2018-06-28 | Japan | Poland | 0 | 1 | 2 | group | 2 | 0.2429 | 0.2428 | 0.5143 | 1.0700 | 1.6720 | True |
| 2018-06-28 | Panama | Tunisia | 1 | 2 | 2 | group | 2 | 0.3282 | 0.2785 | 0.3933 | 1.1140 | 1.2470 | True |
| 2018-06-28 | Senegal | Colombia | 0 | 1 | 2 | group | 2 | 0.2649 | 0.2825 | 0.4526 | 0.9250 | 1.3020 | True |
| 2018-06-30 | France | Argentina | 4 | 3 | 1 | round_of_16 | 1 | 0.3757 | 0.2635 | 0.3608 | 1.3190 | 1.2870 | True |
| 2018-06-30 | Uruguay | Portugal | 2 | 1 | 1 | round_of_16 | 2 | 0.3523 | 0.2776 | 0.3702 | 1.1740 | 1.2110 | False |
| 2018-07-01 | Croatia | Denmark | 1 | 1 | X | round_of_16 | 2 | 0.3349 | 0.2939 | 0.3712 | 1.0460 | 1.1180 | False |
| 2018-07-01 | Russia | Spain | 1 | 1 | X | round_of_16 | 2 | 0.1501 | 0.2286 | 0.6213 | 0.7290 | 1.7620 | False |
| 2018-07-02 | Belgium | Japan | 3 | 2 | 1 | round_of_16 | 1 | 0.6738 | 0.1938 | 0.1324 | 2.1110 | 0.8110 | True |
| 2018-07-02 | Brazil | Mexico | 2 | 0 | 1 | round_of_16 | 1 | 0.6549 | 0.2272 | 0.1180 | 1.7240 | 0.5710 | True |
| 2018-07-03 | Colombia | England | 1 | 1 | X | round_of_16 | 2 | 0.2819 | 0.2922 | 0.4258 | 0.9280 | 1.2120 | False |
| 2018-07-03 | Sweden | Switzerland | 1 | 0 | 1 | round_of_16 | 2 | 0.2861 | 0.3004 | 0.4135 | 0.9040 | 1.1500 | False |
| 2018-07-06 | Brazil | Belgium | 1 | 2 | 2 | quarter_final | 1 | 0.6604 | 0.2028 | 0.1368 | 2.0110 | 0.7850 | False |
| 2018-07-06 | Uruguay | France | 0 | 2 | 2 | quarter_final | 2 | 0.2773 | 0.2698 | 0.4529 | 1.0260 | 1.3910 | True |
| 2018-07-07 | Russia | Croatia | 1 | 1 | X | quarter_final | 1 | 0.3568 | 0.2891 | 0.3541 | 1.1170 | 1.1110 | False |
| 2018-07-07 | Sweden | England | 0 | 2 | 2 | quarter_final | 2 | 0.2107 | 0.2927 | 0.4966 | 0.7210 | 1.2740 | True |
| 2018-07-10 | France | Belgium | 1 | 0 | 1 | semi_final | 1 | 0.4319 | 0.2499 | 0.3182 | 1.5320 | 1.2800 | True |
| 2018-07-11 | Croatia | England | 1 | 1 | X | semi_final | 2 | 0.2090 | 0.2764 | 0.5146 | 0.7730 | 1.3860 | False |
| 2018-07-14 | Belgium | England | 2 | 0 | 1 | third_place | 2 | 0.3150 | 0.2742 | 0.4109 | 1.1060 | 1.3050 | False |
| 2018-07-15 | France | Croatia | 4 | 2 | 1 | final | 1 | 0.5620 | 0.2443 | 0.1938 | 1.6520 | 0.8570 | True |
