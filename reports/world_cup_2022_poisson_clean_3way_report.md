# World Cup 2022 Poisson goal baseline Report (clean_3way)

## Data

- Train rows: 45524
- Train date range: 1872-11-30 to 2022-11-19
- Test rows: 64
- Test date range: 2022-11-20 to 2022-12-18

## Headline Metrics

| Model | Accuracy | Log loss | Brier |
| --- | ---: | ---: | ---: |
| Majority baseline | 0.438 | 1.074 | 0.651 |
| Poisson goal baseline | 0.547 | 1.055 | 0.608 |

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
| 1 | 0.424 |
| X | 0.225 |
| 2 | 0.351 |

## Highest-Confidence Misses

| date | home_team | away_team | home_goals | away_goals | result | world_cup_stage | predicted | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-12-09 | Croatia | Brazil | 0 | 0 | X | quarter_final | 2 | 0.927 |
| 2022-12-02 | Cameroon | Brazil | 1 | 0 | 1 | group | 2 | 0.885 |
| 2022-12-09 | Netherlands | Argentina | 2 | 2 | X | quarter_final | 2 | 0.839 |
| 2022-12-10 | Morocco | Portugal | 1 | 0 | 1 | quarter_final | 2 | 0.833 |
| 2022-11-22 | Argentina | Saudi Arabia | 1 | 2 | 2 | group | 1 | 0.734 |
| 2022-12-18 | Argentina | France | 2 | 2 | X | final | 1 | 0.716 |
| 2022-11-30 | Tunisia | France | 1 | 0 | 1 | group | 2 | 0.604 |
| 2022-12-01 | Japan | Spain | 2 | 1 | 1 | group | 2 | 0.601 |
| 2022-11-28 | South Korea | Ghana | 2 | 3 | 2 | group | 1 | 0.575 |
| 2022-12-02 | South Korea | Portugal | 2 | 1 | 1 | group | 2 | 0.567 |

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
| 2022-11-25 | England | United States | 0 | 0 | X | group | 1 | 0.4764 | 0.2628 | 0.2608 | 1.4670 | 1.0130 | False |
| 2022-11-25 | Netherlands | Ecuador | 1 | 1 | X | group | 1 | 0.5438 | 0.2548 | 0.2013 | 1.5590 | 0.8370 | False |
| 2022-11-25 | Qatar | Senegal | 1 | 3 | 2 | group | 1 | 0.4412 | 0.2806 | 0.2782 | 1.3020 | 0.9720 | False |
| 2022-11-25 | Wales | Iran | 0 | 2 | 2 | group | 2 | 0.2774 | 0.2731 | 0.4496 | 1.0080 | 1.3630 | True |
| 2022-11-26 | Argentina | Mexico | 2 | 0 | 1 | group | 1 | 0.6737 | 0.2112 | 0.1152 | 1.8650 | 0.6150 | True |
| 2022-11-26 | France | Denmark | 2 | 1 | 1 | group | 1 | 0.3968 | 0.2629 | 0.3403 | 1.3640 | 1.2430 | True |
| 2022-11-26 | Poland | Saudi Arabia | 2 | 0 | 1 | group | 1 | 0.4713 | 0.2821 | 0.2466 | 1.3220 | 0.8720 | True |
| 2022-11-26 | Tunisia | Australia | 0 | 1 | 2 | group | 1 | 0.3698 | 0.2787 | 0.3514 | 1.2030 | 1.1650 | False |
| 2022-11-27 | Belgium | Morocco | 0 | 2 | 2 | group | 1 | 0.5424 | 0.2376 | 0.2200 | 1.7270 | 1.0090 | False |
| 2022-11-27 | Croatia | Canada | 4 | 1 | 1 | group | 1 | 0.4902 | 0.2458 | 0.2640 | 1.6310 | 1.1300 | True |
| 2022-11-27 | Japan | Costa Rica | 0 | 1 | 2 | group | 1 | 0.4469 | 0.2772 | 0.2759 | 1.3310 | 0.9820 | False |
| 2022-11-27 | Spain | Germany | 1 | 1 | X | group | 1 | 0.4502 | 0.2511 | 0.2988 | 1.5450 | 1.2120 | False |
| 2022-11-28 | Brazil | Switzerland | 1 | 0 | 1 | group | 1 | 0.7189 | 0.1827 | 0.0984 | 2.1200 | 0.6350 | True |
| 2022-11-28 | Cameroon | Serbia | 3 | 3 | X | group | 2 | 0.1992 | 0.2407 | 0.5601 | 0.9000 | 1.6890 | False |
| 2022-11-28 | Portugal | Uruguay | 2 | 0 | 1 | group | 1 | 0.3973 | 0.2750 | 0.3277 | 1.2770 | 1.1330 | True |
| 2022-11-28 | South Korea | Ghana | 2 | 3 | 2 | group | 1 | 0.5752 | 0.2469 | 0.1779 | 1.6200 | 0.7760 | False |
| 2022-11-29 | Ecuador | Senegal | 1 | 2 | 2 | group | 1 | 0.4149 | 0.3027 | 0.2823 | 1.1400 | 0.8850 | False |
| 2022-11-29 | Iran | United States | 0 | 1 | 2 | group | 2 | 0.3227 | 0.2907 | 0.3865 | 1.0350 | 1.1620 | True |
| 2022-11-29 | Qatar | Netherlands | 0 | 2 | 2 | group | 2 | 0.1499 | 0.2391 | 0.6111 | 0.6830 | 1.6670 | True |
| 2022-11-29 | Wales | England | 0 | 3 | 2 | group | 2 | 0.1405 | 0.2263 | 0.6332 | 0.6900 | 1.7700 | True |
| 2022-11-30 | Australia | Denmark | 1 | 0 | 1 | group | 2 | 0.1959 | 0.2545 | 0.5496 | 0.8170 | 1.5610 | False |
| 2022-11-30 | Poland | Argentina | 0 | 2 | 2 | group | 2 | 0.0671 | 0.1584 | 0.7744 | 0.5020 | 2.2550 | True |
| 2022-11-30 | Saudi Arabia | Mexico | 1 | 2 | 2 | group | 2 | 0.2111 | 0.3043 | 0.4846 | 0.6860 | 1.2030 | True |
| 2022-11-30 | Tunisia | France | 1 | 0 | 1 | group | 2 | 0.1562 | 0.2399 | 0.6039 | 0.7090 | 1.6660 | False |
| 2022-12-01 | Canada | Morocco | 1 | 2 | 2 | group | 2 | 0.3236 | 0.2658 | 0.4107 | 1.1810 | 1.3660 | True |
| 2022-12-01 | Costa Rica | Germany | 2 | 4 | 2 | group | 2 | 0.1651 | 0.2426 | 0.5923 | 0.7380 | 1.6500 | True |
| 2022-12-01 | Croatia | Belgium | 0 | 0 | X | group | 2 | 0.2409 | 0.2480 | 0.5111 | 1.0290 | 1.6190 | False |
| 2022-12-01 | Japan | Spain | 2 | 1 | 1 | group | 2 | 0.1656 | 0.2336 | 0.6008 | 0.7820 | 1.7340 | False |
| 2022-12-02 | Cameroon | Brazil | 1 | 0 | 1 | group | 2 | 0.0240 | 0.0910 | 0.8850 | 0.3180 | 2.8330 | False |
| 2022-12-02 | Ghana | Uruguay | 0 | 2 | 2 | group | 2 | 0.0977 | 0.2127 | 0.6896 | 0.5100 | 1.8060 | True |
| 2022-12-02 | Serbia | Switzerland | 2 | 3 | 2 | group | 2 | 0.3159 | 0.2641 | 0.4200 | 1.1710 | 1.3930 | True |
| 2022-12-02 | South Korea | Portugal | 2 | 1 | 1 | group | 2 | 0.1903 | 0.2427 | 0.5670 | 0.8500 | 1.6650 | False |
| 2022-12-03 | Argentina | Australia | 2 | 1 | 1 | round_of_16 | 1 | 0.8571 | 0.1002 | 0.0427 | 3.0100 | 0.5650 | True |
| 2022-12-03 | Netherlands | United States | 3 | 1 | 1 | round_of_16 | 1 | 0.6976 | 0.1764 | 0.1259 | 2.3340 | 0.8850 | True |
| 2022-12-04 | England | Senegal | 3 | 0 | 1 | round_of_16 | 1 | 0.7304 | 0.1620 | 0.1075 | 2.4690 | 0.8410 | True |
| 2022-12-04 | France | Poland | 3 | 1 | 1 | round_of_16 | 1 | 0.7042 | 0.1742 | 0.1216 | 2.3480 | 0.8670 | True |
| 2022-12-05 | Brazil | South Korea | 4 | 1 | 1 | round_of_16 | 1 | 0.8491 | 0.1037 | 0.0472 | 3.0060 | 0.6070 | True |
| 2022-12-05 | Japan | Croatia | 1 | 1 | X | round_of_16 | 1 | 0.5033 | 0.2307 | 0.2660 | 1.8060 | 1.2550 | False |
| 2022-12-06 | Morocco | Spain | 0 | 0 | X | round_of_16 | 2 | 0.2496 | 0.2354 | 0.5150 | 1.1500 | 1.7530 | False |
| 2022-12-06 | Portugal | Switzerland | 6 | 1 | 1 | round_of_16 | 1 | 0.6640 | 0.1917 | 0.1443 | 2.1780 | 0.9060 | True |
| 2022-12-09 | Croatia | Brazil | 0 | 0 | X | quarter_final | 2 | 0.0124 | 0.0609 | 0.9267 | 0.2460 | 3.2290 | False |
| 2022-12-09 | Netherlands | Argentina | 2 | 2 | X | quarter_final | 2 | 0.0398 | 0.1214 | 0.8388 | 0.3940 | 2.5390 | False |
| 2022-12-10 | England | France | 1 | 2 | 2 | quarter_final | 2 | 0.0737 | 0.1786 | 0.7477 | 0.4730 | 2.0540 | True |
| 2022-12-10 | Morocco | Portugal | 1 | 0 | 1 | quarter_final | 2 | 0.0391 | 0.1282 | 0.8327 | 0.3590 | 2.4280 | False |
| 2022-12-13 | Argentina | Croatia | 3 | 0 | 1 | semi_final | 1 | 0.7526 | 0.2006 | 0.0467 | 1.7540 | 0.2490 | True |
| 2022-12-14 | France | Morocco | 2 | 0 | 1 | semi_final | 1 | 0.6565 | 0.2640 | 0.0795 | 1.4230 | 0.3130 | True |
| 2022-12-17 | Croatia | Morocco | 2 | 1 | 1 | third_place | 1 | 0.8937 | 0.1048 | 0.0014 | 2.2730 | 0.0140 | True |
| 2022-12-18 | Argentina | France | 2 | 2 | X | final | 1 | 0.7159 | 0.1309 | 0.1532 | 3.6860 | 1.8390 | False |
