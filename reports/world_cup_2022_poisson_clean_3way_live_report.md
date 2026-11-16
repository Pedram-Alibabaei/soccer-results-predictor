# World Cup 2022 Poisson goal baseline Report (clean_3way, live tournament ratings)

## Data

- Train rows: 45524
- Train date range: 1872-11-30 to 2022-11-19
- Test rows: 64
- Test date range: 2022-11-20 to 2022-12-18

## Headline Metrics

| Model | Accuracy | Log loss | Brier |
| --- | ---: | ---: | ---: |
| Majority baseline | 0.438 | 1.074 | 0.651 |
| Poisson goal baseline | 0.547 | 1.059 | 0.610 |

## Confusion Matrix

Rows are actual results; columns are predicted results.

| actual | 1 | X | 2 |
| --- | --- | --- | --- |
| 1 | 22 | 0 | 6 |
| X | 9 | 0 | 6 |
| 2 | 8 | 0 | 13 |

## Predicted Result Distribution

| result | count |
| --- | --- |
| 1 | 39 |
| X | 0 |
| 2 | 25 |

## Average Predicted Probabilities

| result | probability |
| --- | --- |
| 1 | 0.420 |
| X | 0.229 |
| 2 | 0.351 |

## Highest-Confidence Misses

| date | home_team | away_team | home_goals | away_goals | result | world_cup_stage | predicted | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-12-09 | Croatia | Brazil | 0 | 0 | X | quarter_final | 2 | 0.903 |
| 2022-12-02 | Cameroon | Brazil | 1 | 0 | 1 | group | 2 | 0.902 |
| 2022-12-09 | Netherlands | Argentina | 2 | 2 | X | quarter_final | 2 | 0.801 |
| 2022-12-10 | Morocco | Portugal | 1 | 0 | 1 | quarter_final | 2 | 0.790 |
| 2022-11-22 | Argentina | Saudi Arabia | 1 | 2 | 2 | group | 1 | 0.734 |
| 2022-11-30 | Tunisia | France | 1 | 0 | 1 | group | 2 | 0.628 |
| 2022-12-18 | Argentina | France | 2 | 2 | X | final | 1 | 0.627 |
| 2022-12-01 | Japan | Spain | 2 | 1 | 1 | group | 2 | 0.621 |
| 2022-12-02 | South Korea | Portugal | 2 | 1 | 1 | group | 2 | 0.616 |
| 2022-11-28 | South Korea | Ghana | 2 | 3 | 2 | group | 1 | 0.608 |

## Match Predictions

| date | home_team | away_team | home_goals | away_goals | result | world_cup_stage | predicted | p_1 | p_X | p_2 | home_xg | away_xg | correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-11-20 | Qatar | Ecuador | 0 | 2 | 2 | group | 2 | 0.3267 | 0.3188 | 0.3545 | 0.9210 | 0.9720 | True |
| 2022-11-21 | England | Iran | 6 | 2 | 1 | group | 1 | 0.4648 | 0.2707 | 0.2646 | 1.3960 | 0.9820 | True |
| 2022-11-21 | Senegal | Netherlands | 0 | 2 | 2 | group | 2 | 0.1260 | 0.2127 | 0.6614 | 0.6710 | 1.8750 | True |
| 2022-11-21 | United States | Wales | 1 | 1 | X | group | 1 | 0.4478 | 0.2890 | 0.2632 | 1.2560 | 0.8910 | False |
| 2022-11-22 | Argentina | Saudi Arabia | 1 | 2 | 2 | group | 1 | 0.7339 | 0.1959 | 0.0702 | 1.8710 | 0.3970 | False |
| 2022-11-22 | Denmark | Tunisia | 0 | 0 | X | group | 1 | 0.5216 | 0.2645 | 0.2138 | 1.4760 | 0.8400 | False |
| 2022-11-22 | France | Australia | 4 | 1 | 1 | group | 1 | 0.5548 | 0.2519 | 0.1933 | 1.5830 | 0.8180 | True |
| 2022-11-22 | Mexico | Poland | 0 | 0 | X | group | 1 | 0.3844 | 0.2857 | 0.3299 | 1.1880 | 1.0780 | False |
| 2022-11-23 | Belgium | Canada | 1 | 0 | 1 | group | 1 | 0.5551 | 0.2372 | 0.2077 | 1.7280 | 0.9570 | True |
| 2022-11-23 | Germany | Japan | 1 | 2 | 2 | group | 1 | 0.4858 | 0.2568 | 0.2574 | 1.5250 | 1.0360 | False |
| 2022-11-23 | Morocco | Croatia | 0 | 0 | X | group | 2 | 0.3188 | 0.2720 | 0.4091 | 1.1290 | 1.3170 | False |
| 2022-11-23 | Spain | Costa Rica | 7 | 0 | 1 | group | 1 | 0.6093 | 0.2411 | 0.1496 | 1.6490 | 0.6730 | True |
| 2022-11-24 | Brazil | Serbia | 2 | 0 | 1 | group | 1 | 0.7623 | 0.1527 | 0.0850 | 2.4680 | 0.6970 | True |
| 2022-11-24 | Portugal | Ghana | 3 | 2 | 1 | group | 1 | 0.7020 | 0.2009 | 0.0971 | 1.9150 | 0.5480 | True |
| 2022-11-24 | Switzerland | Cameroon | 1 | 0 | 1 | group | 1 | 0.5651 | 0.2553 | 0.1796 | 1.5500 | 0.7470 | True |
| 2022-11-24 | Uruguay | South Korea | 0 | 0 | X | group | 1 | 0.4594 | 0.2833 | 0.2573 | 1.3040 | 0.8990 | False |
| 2022-11-25 | England | United States | 0 | 0 | X | group | 1 | 0.4983 | 0.2532 | 0.2485 | 1.5640 | 1.0260 | False |
| 2022-11-25 | Netherlands | Ecuador | 1 | 1 | X | group | 1 | 0.5353 | 0.2574 | 0.2073 | 1.5370 | 0.8480 | False |
| 2022-11-25 | Qatar | Senegal | 1 | 3 | 2 | group | 1 | 0.4278 | 0.2835 | 0.2887 | 1.2670 | 0.9870 | False |
| 2022-11-25 | Wales | Iran | 0 | 2 | 2 | group | 2 | 0.3309 | 0.2676 | 0.4015 | 1.1880 | 1.3370 | True |
| 2022-11-26 | Argentina | Mexico | 2 | 0 | 1 | group | 1 | 0.6238 | 0.2346 | 0.1416 | 1.6960 | 0.6620 | True |
| 2022-11-26 | France | Denmark | 2 | 1 | 1 | group | 1 | 0.4139 | 0.2632 | 0.3229 | 1.3900 | 1.1960 | True |
| 2022-11-26 | Poland | Saudi Arabia | 2 | 0 | 1 | group | 1 | 0.4521 | 0.2863 | 0.2616 | 1.2770 | 0.8980 | True |
| 2022-11-26 | Tunisia | Australia | 0 | 1 | 2 | group | 1 | 0.4269 | 0.2729 | 0.3002 | 1.3360 | 1.0740 | False |
| 2022-11-27 | Belgium | Morocco | 0 | 2 | 2 | group | 1 | 0.5462 | 0.2427 | 0.2111 | 1.6730 | 0.9400 | False |
| 2022-11-27 | Croatia | Canada | 4 | 1 | 1 | group | 1 | 0.5006 | 0.2500 | 0.2494 | 1.5960 | 1.0490 | True |
| 2022-11-27 | Japan | Costa Rica | 0 | 1 | 2 | group | 1 | 0.5261 | 0.2552 | 0.2187 | 1.5550 | 0.9030 | False |
| 2022-11-27 | Spain | Germany | 1 | 1 | X | group | 1 | 0.4975 | 0.2404 | 0.2621 | 1.6920 | 1.1620 | False |
| 2022-11-28 | Brazil | Switzerland | 1 | 0 | 1 | group | 1 | 0.7083 | 0.1899 | 0.1017 | 2.0480 | 0.6230 | True |
| 2022-11-28 | Cameroon | Serbia | 3 | 3 | X | group | 2 | 0.2076 | 0.2474 | 0.5450 | 0.8990 | 1.6270 | False |
| 2022-11-28 | Portugal | Uruguay | 2 | 0 | 1 | group | 1 | 0.3897 | 0.2757 | 0.3346 | 1.2590 | 1.1460 | True |
| 2022-11-28 | South Korea | Ghana | 2 | 3 | 2 | group | 1 | 0.6080 | 0.2348 | 0.1572 | 1.7130 | 0.7360 | False |
| 2022-11-29 | Ecuador | Senegal | 1 | 2 | 2 | group | 1 | 0.4433 | 0.2919 | 0.2647 | 1.2340 | 0.8830 | False |
| 2022-11-29 | Iran | United States | 0 | 1 | 2 | group | 2 | 0.2969 | 0.2813 | 0.4217 | 1.0200 | 1.2730 | True |
| 2022-11-29 | Qatar | Netherlands | 0 | 2 | 2 | group | 2 | 0.1125 | 0.2150 | 0.6725 | 0.5850 | 1.8210 | True |
| 2022-11-29 | Wales | England | 0 | 3 | 2 | group | 2 | 0.1198 | 0.2130 | 0.6671 | 0.6340 | 1.8580 | True |
| 2022-11-30 | Australia | Denmark | 1 | 0 | 1 | group | 2 | 0.1934 | 0.2544 | 0.5522 | 0.8070 | 1.5620 | False |
| 2022-11-30 | Poland | Argentina | 0 | 2 | 2 | group | 2 | 0.0920 | 0.1935 | 0.7144 | 0.5440 | 1.9730 | True |
| 2022-11-30 | Saudi Arabia | Mexico | 1 | 2 | 2 | group | 2 | 0.2130 | 0.3019 | 0.4850 | 0.6990 | 1.2150 | True |
| 2022-11-30 | Tunisia | France | 1 | 0 | 1 | group | 2 | 0.1379 | 0.2343 | 0.6278 | 0.6440 | 1.6940 | False |
| 2022-12-01 | Canada | Morocco | 1 | 2 | 2 | group | 2 | 0.2333 | 0.2568 | 0.5099 | 0.9490 | 1.5370 | True |
| 2022-12-01 | Costa Rica | Germany | 2 | 4 | 2 | group | 2 | 0.1478 | 0.2279 | 0.6243 | 0.7200 | 1.7660 | True |
| 2022-12-01 | Croatia | Belgium | 0 | 0 | X | group | 2 | 0.2755 | 0.2618 | 0.4627 | 1.0660 | 1.4640 | False |
| 2022-12-01 | Japan | Spain | 2 | 1 | 1 | group | 2 | 0.1531 | 0.2259 | 0.6210 | 0.7570 | 1.7940 | False |
| 2022-12-02 | Cameroon | Brazil | 1 | 0 | 1 | group | 2 | 0.0196 | 0.0785 | 0.9019 | 0.3050 | 3.0040 | False |
| 2022-12-02 | Ghana | Uruguay | 0 | 2 | 2 | group | 2 | 0.1088 | 0.2114 | 0.6799 | 0.5780 | 1.8470 | True |
| 2022-12-02 | Serbia | Switzerland | 2 | 3 | 2 | group | 2 | 0.2688 | 0.2567 | 0.4745 | 1.0750 | 1.5170 | True |
| 2022-12-02 | South Korea | Portugal | 2 | 1 | 1 | group | 2 | 0.1573 | 0.2263 | 0.6164 | 0.7760 | 1.7950 | False |
| 2022-12-03 | Argentina | Australia | 2 | 1 | 1 | round_of_16 | 1 | 0.8326 | 0.1150 | 0.0524 | 2.8310 | 0.5930 | True |
| 2022-12-03 | Netherlands | United States | 3 | 1 | 1 | round_of_16 | 1 | 0.6749 | 0.1921 | 0.1330 | 2.1350 | 0.8260 | True |
| 2022-12-04 | England | Senegal | 3 | 0 | 1 | round_of_16 | 1 | 0.7354 | 0.1584 | 0.1062 | 2.5260 | 0.8590 | True |
| 2022-12-04 | France | Poland | 3 | 1 | 1 | round_of_16 | 1 | 0.6735 | 0.1903 | 0.1362 | 2.1720 | 0.8610 | True |
| 2022-12-05 | Brazil | South Korea | 4 | 1 | 1 | round_of_16 | 1 | 0.8330 | 0.1147 | 0.0523 | 2.8360 | 0.5940 | True |
| 2022-12-05 | Japan | Croatia | 1 | 1 | X | round_of_16 | 1 | 0.4848 | 0.2363 | 0.2788 | 1.7290 | 1.2580 | False |
| 2022-12-06 | Morocco | Spain | 0 | 0 | X | round_of_16 | 2 | 0.3317 | 0.2450 | 0.4233 | 1.3610 | 1.5670 | False |
| 2022-12-06 | Portugal | Switzerland | 6 | 1 | 1 | round_of_16 | 1 | 0.6255 | 0.2031 | 0.1714 | 2.0940 | 0.9980 | True |
| 2022-12-09 | Croatia | Brazil | 0 | 0 | X | quarter_final | 2 | 0.0170 | 0.0801 | 0.9029 | 0.2500 | 2.8990 | False |
| 2022-12-09 | Netherlands | Argentina | 2 | 2 | X | quarter_final | 2 | 0.0522 | 0.1471 | 0.8007 | 0.4160 | 2.2930 | False |
| 2022-12-10 | England | France | 1 | 2 | 2 | quarter_final | 2 | 0.0884 | 0.1936 | 0.7180 | 0.5200 | 1.9590 | True |
| 2022-12-10 | Morocco | Portugal | 1 | 0 | 1 | quarter_final | 2 | 0.0543 | 0.1558 | 0.7899 | 0.4030 | 2.2000 | False |
| 2022-12-13 | Argentina | Croatia | 3 | 0 | 1 | semi_final | 1 | 0.6979 | 0.2388 | 0.0634 | 1.5450 | 0.2790 | True |
| 2022-12-14 | France | Morocco | 2 | 0 | 1 | semi_final | 1 | 0.6088 | 0.2938 | 0.0974 | 1.2800 | 0.3390 | True |
| 2022-12-17 | Croatia | Morocco | 2 | 1 | 1 | third_place | 1 | 0.8464 | 0.1514 | 0.0022 | 1.9010 | 0.0140 | True |
| 2022-12-18 | Argentina | France | 2 | 2 | X | final | 1 | 0.6272 | 0.1508 | 0.2219 | 3.4550 | 2.1690 | False |
