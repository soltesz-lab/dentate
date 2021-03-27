export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=yes
ml load intel19

fil=(
" HC 1030000 28682721 results/netclamp/network_clamp.optimize.HC_1030000_20210204_025428_28682721.yaml "
" HC 1030000 15879716 results/netclamp/network_clamp.optimize.HC_1030000_20210204_035942_15879716.yaml "
" HC 1030000 45419272 results/netclamp/network_clamp.optimize.HC_1030000_20210204_035942_45419272.yaml "
" HC 1030000 53736785 results/netclamp/network_clamp.optimize.HC_1030000_20210204_035942_53736785.yaml "
" HC 1030000 63599789 results/netclamp/network_clamp.optimize.HC_1030000_20210204_035942_63599789.yaml "
" HC 1032250 204284 results/netclamp/network_clamp.optimize.HC_1032250_20210204_025428_00204284.yaml "
" HC 1032250 13571683 results/netclamp/network_clamp.optimize.HC_1032250_20210204_025428_13571683.yaml "
" HC 1032250 29585629 results/netclamp/network_clamp.optimize.HC_1032250_20210204_025428_29585629.yaml "
" HC 1032250 80350627 results/netclamp/network_clamp.optimize.HC_1032250_20210204_025428_80350627.yaml "
" HC 1032250 39786206 results/netclamp/network_clamp.optimize.HC_1032250_20210204_035942_39786206.yaml "
" HC 1034500 17324444 results/netclamp/network_clamp.optimize.HC_1034500_20210204_025430_17324444.yaml "
" HC 1034500 19663111 results/netclamp/network_clamp.optimize.HC_1034500_20210204_025430_19663111.yaml "
" HC 1034500 48663281 results/netclamp/network_clamp.optimize.HC_1034500_20210204_025430_48663281.yaml "
" HC 1034500 61645166 results/netclamp/network_clamp.optimize.HC_1034500_20210204_025430_61645166.yaml "
" HC 1034500 85653587 results/netclamp/network_clamp.optimize.HC_1034500_20210204_025430_85653587.yaml "
" HC 1036750 9389064 results/netclamp/network_clamp.optimize.HC_1036750_20210204_025429_09389064.yaml "
" HC 1036750 72215563 results/netclamp/network_clamp.optimize.HC_1036750_20210204_025429_72215563.yaml "
" HC 1036750 86292756 results/netclamp/network_clamp.optimize.HC_1036750_20210204_025429_86292756.yaml "
" HC 1036750 96494881 results/netclamp/network_clamp.optimize.HC_1036750_20210204_025429_96494881.yaml "
" HC 1036750 96962730 results/netclamp/network_clamp.optimize.HC_1036750_20210204_025430_96962730.yaml "
" HC 1038999 19285510 results/netclamp/network_clamp.optimize.HC_1038999_20210204_025429_19285510.yaml "
" HC 1038999 45197076 results/netclamp/network_clamp.optimize.HC_1038999_20210204_025429_45197076.yaml "
" HC 1038999 57153876 results/netclamp/network_clamp.optimize.HC_1038999_20210204_025429_57153876.yaml "
" HC 1038999 58867820 results/netclamp/network_clamp.optimize.HC_1038999_20210204_025429_58867820.yaml "
" HC 1038999 69758687 results/netclamp/network_clamp.optimize.HC_1038999_20210204_025429_69758687.yaml "
" BC 1039000 1503844 results/netclamp/network_clamp.optimize.BC_1039000_20210204_025430_01503844.yaml "
" BC 1039000 28135771 results/netclamp/network_clamp.optimize.BC_1039000_20210204_025430_28135771.yaml "
" BC 1039000 52357252 results/netclamp/network_clamp.optimize.BC_1039000_20210204_025430_52357252.yaml "
" BC 1039000 74865768 results/netclamp/network_clamp.optimize.BC_1039000_20210204_025430_74865768.yaml "
" BC 1039000 93454042 results/netclamp/network_clamp.optimize.BC_1039000_20210204_025430_93454042.yaml "
" BC 1039950 27431042 results/netclamp/network_clamp.optimize.BC_1039950_20210204_025430_27431042.yaml "
" BC 1039950 49779904 results/netclamp/network_clamp.optimize.BC_1039950_20210204_025430_49779904.yaml "
" BC 1039950 67539344 results/netclamp/network_clamp.optimize.BC_1039950_20210204_025430_67539344.yaml "
" BC 1039950 90485672 results/netclamp/network_clamp.optimize.BC_1039950_20210204_025430_90485672.yaml "
" BC 1039950 97486157 results/netclamp/network_clamp.optimize.BC_1039950_20210204_025430_97486157.yaml "
" BC 1040900 4643442 results/netclamp/network_clamp.optimize.BC_1040900_20210204_025430_04643442.yaml "
" BC 1040900 32603857 results/netclamp/network_clamp.optimize.BC_1040900_20210204_025430_32603857.yaml "
" BC 1040900 63255735 results/netclamp/network_clamp.optimize.BC_1040900_20210204_025430_63255735.yaml "
" BC 1040900 97616150 results/netclamp/network_clamp.optimize.BC_1040900_20210204_025430_97616150.yaml "
" BC 1040900 98230850 results/netclamp/network_clamp.optimize.BC_1040900_20210204_025430_98230850.yaml "
" BC 1041850 19191189 results/netclamp/network_clamp.optimize.BC_1041850_20210204_025430_19191189.yaml "
" BC 1041850 21074287 results/netclamp/network_clamp.optimize.BC_1041850_20210204_025430_21074287.yaml "
" BC 1041850 63875616 results/netclamp/network_clamp.optimize.BC_1041850_20210204_025430_63875616.yaml "
" BC 1041850 67275914 results/netclamp/network_clamp.optimize.BC_1041850_20210204_025430_67275914.yaml "
" BC 1041850 87742037 results/netclamp/network_clamp.optimize.BC_1041850_20210204_025430_87742037.yaml "
" BC 1042799 36739208 results/netclamp/network_clamp.optimize.BC_1042799_20210204_025429_36739208.yaml "
" BC 1042799 57586813 results/netclamp/network_clamp.optimize.BC_1042799_20210204_025429_57586813.yaml "
" BC 1042799 62574877 results/netclamp/network_clamp.optimize.BC_1042799_20210204_025429_62574877.yaml "
" BC 1042799 183453 results/netclamp/network_clamp.optimize.BC_1042799_20210204_025430_00183453.yaml "
" BC 1042799 67308587 results/netclamp/network_clamp.optimize.BC_1042799_20210204_035942_67308587.yaml "
" AAC 1042800 4893658 results/netclamp/network_clamp.optimize.AAC_1042800_20210204_025429_04893658.yaml "
" AAC 1042800 27137089 results/netclamp/network_clamp.optimize.AAC_1042800_20210204_025429_27137089.yaml "
" AAC 1042800 36010476 results/netclamp/network_clamp.optimize.AAC_1042800_20210204_025429_36010476.yaml "
" AAC 1042800 49937004 results/netclamp/network_clamp.optimize.AAC_1042800_20210204_025429_49937004.yaml "
" AAC 1042800 53499406 results/netclamp/network_clamp.optimize.AAC_1042800_20210204_025429_53499406.yaml "
" AAC 1042913 8379552 results/netclamp/network_clamp.optimize.AAC_1042913_20210204_025428_08379552.yaml "
" AAC 1042913 9111622 results/netclamp/network_clamp.optimize.AAC_1042913_20210204_025428_09111622.yaml "
" AAC 1042913 38840365 results/netclamp/network_clamp.optimize.AAC_1042913_20210204_025428_38840365.yaml "
" AAC 1042913 80515073 results/netclamp/network_clamp.optimize.AAC_1042913_20210204_025428_80515073.yaml "
" AAC 1042913 291547 results/netclamp/network_clamp.optimize.AAC_1042913_20210204_025429_00291547.yaml "
" AAC 1043025 52829774 results/netclamp/network_clamp.optimize.AAC_1043025_20210204_025424_52829774.yaml "
" AAC 1043025 62387369 results/netclamp/network_clamp.optimize.AAC_1043025_20210204_025427_62387369.yaml "
" AAC 1043025 7818268 results/netclamp/network_clamp.optimize.AAC_1043025_20210204_025428_07818268.yaml "
" AAC 1043025 59206615 results/netclamp/network_clamp.optimize.AAC_1043025_20210204_025428_59206615.yaml "
" AAC 1043025 82956063 results/netclamp/network_clamp.optimize.AAC_1043025_20210204_025429_82956063.yaml "
" AAC 1043138 19281943 results/netclamp/network_clamp.optimize.AAC_1043138_20210204_025428_19281943.yaml "
" AAC 1043138 40133402 results/netclamp/network_clamp.optimize.AAC_1043138_20210204_025430_40133402.yaml "
" AAC 1043138 70337332 results/netclamp/network_clamp.optimize.AAC_1043138_20210204_025430_70337332.yaml "
" AAC 1043138 82470709 results/netclamp/network_clamp.optimize.AAC_1043138_20210204_025430_82470709.yaml "
" AAC 1043138 85264434 results/netclamp/network_clamp.optimize.AAC_1043138_20210204_025430_85264434.yaml "
" AAC 1043249 66598438 results/netclamp/network_clamp.optimize.AAC_1043249_20210204_025429_66598438.yaml "
" AAC 1043249 95905199 results/netclamp/network_clamp.optimize.AAC_1043249_20210204_025429_95905199.yaml "
" AAC 1043249 43400137 results/netclamp/network_clamp.optimize.AAC_1043249_20210204_025430_43400137.yaml "
" AAC 1043249 54652217 results/netclamp/network_clamp.optimize.AAC_1043249_20210204_025430_54652217.yaml "
" AAC 1043249 26662642 results/netclamp/network_clamp.optimize.AAC_1043249_20210204_035942_26662642.yaml "
" HCC 1043250 12260638 results/netclamp/network_clamp.optimize.HCC_1043250_20210204_025428_12260638.yaml "
" HCC 1043250 17609813 results/netclamp/network_clamp.optimize.HCC_1043250_20210204_025428_17609813.yaml "
" HCC 1043250 71407528 results/netclamp/network_clamp.optimize.HCC_1043250_20210204_025428_71407528.yaml "
" HCC 1043250 33236209 results/netclamp/network_clamp.optimize.HCC_1043250_20210204_025429_33236209.yaml "
" HCC 1043250 92055940 results/netclamp/network_clamp.optimize.HCC_1043250_20210204_025429_92055940.yaml "
" HCC 1043600 17293770 results/netclamp/network_clamp.optimize.HCC_1043600_20210204_025430_17293770.yaml "
" HCC 1043600 60991105 results/netclamp/network_clamp.optimize.HCC_1043600_20210204_025430_60991105.yaml "
" HCC 1043600 64290895 results/netclamp/network_clamp.optimize.HCC_1043600_20210204_025430_64290895.yaml "
" HCC 1043600 89433777 results/netclamp/network_clamp.optimize.HCC_1043600_20210204_025430_89433777.yaml "
" HCC 1043600 42402504 results/netclamp/network_clamp.optimize.HCC_1043600_20210204_035942_42402504.yaml "
" HCC 1043950 41566230 results/netclamp/network_clamp.optimize.HCC_1043950_20210204_025430_41566230.yaml "
" HCC 1043950 54593731 results/netclamp/network_clamp.optimize.HCC_1043950_20210204_025430_54593731.yaml "
" HCC 1043950 57660249 results/netclamp/network_clamp.optimize.HCC_1043950_20210204_025430_57660249.yaml "
" HCC 1043950 72125941 results/netclamp/network_clamp.optimize.HCC_1043950_20210204_025430_72125941.yaml "
" HCC 1043950 99434948 results/netclamp/network_clamp.optimize.HCC_1043950_20210204_025430_99434948.yaml "
" HCC 1044300 63592626 results/netclamp/network_clamp.optimize.HCC_1044300_20210204_025429_63592626.yaml "
" HCC 1044300 92910319 results/netclamp/network_clamp.optimize.HCC_1044300_20210204_025429_92910319.yaml "
" HCC 1044300 94613541 results/netclamp/network_clamp.optimize.HCC_1044300_20210204_025429_94613541.yaml "
" HCC 1044300 66834161 results/netclamp/network_clamp.optimize.HCC_1044300_20210204_025430_66834161.yaml "
" HCC 1044300 97569363 results/netclamp/network_clamp.optimize.HCC_1044300_20210204_025430_97569363.yaml "
" HCC 1044649 84121621 results/netclamp/network_clamp.optimize.HCC_1044649_20210204_025429_84121621.yaml "
" HCC 1044649 94560988 results/netclamp/network_clamp.optimize.HCC_1044649_20210204_025429_94560988.yaml "
" HCC 1044649 24805208 results/netclamp/network_clamp.optimize.HCC_1044649_20210204_025430_24805208.yaml "
" HCC 1044649 46366417 results/netclamp/network_clamp.optimize.HCC_1044649_20210204_025430_46366417.yaml "
" HCC 1044649 59396015 results/netclamp/network_clamp.optimize.HCC_1044649_20210204_025430_59396015.yaml "
" NGFC 1044650 12740157 results/netclamp/network_clamp.optimize.NGFC_1044650_20210204_025430_12740157.yaml "
" NGFC 1044650 97895890 results/netclamp/network_clamp.optimize.NGFC_1044650_20210204_025430_97895890.yaml "
" NGFC 1044650 93872787 results/netclamp/network_clamp.optimize.NGFC_1044650_20210204_025431_93872787.yaml "
" NGFC 1044650 95844113 results/netclamp/network_clamp.optimize.NGFC_1044650_20210204_025431_95844113.yaml "
" NGFC 1044650 96772370 results/netclamp/network_clamp.optimize.NGFC_1044650_20210204_025431_96772370.yaml "
" NGFC 1045900 6112188 results/netclamp/network_clamp.optimize.NGFC_1045900_20210204_025429_06112188.yaml "
" NGFC 1045900 71039025 results/netclamp/network_clamp.optimize.NGFC_1045900_20210204_025429_71039025.yaml "
" NGFC 1045900 89814943 results/netclamp/network_clamp.optimize.NGFC_1045900_20210204_025429_89814943.yaml "
" NGFC 1045900 67428613 results/netclamp/network_clamp.optimize.NGFC_1045900_20210204_025431_67428613.yaml "
" NGFC 1045900 95436908 results/netclamp/network_clamp.optimize.NGFC_1045900_20210204_025431_95436908.yaml "
" NGFC 1047150 59071557 results/netclamp/network_clamp.optimize.NGFC_1047150_20210204_025429_59071557.yaml "
" NGFC 1047150 77901687 results/netclamp/network_clamp.optimize.NGFC_1047150_20210204_025429_77901687.yaml "
" NGFC 1047150 27400566 results/netclamp/network_clamp.optimize.NGFC_1047150_20210204_025430_27400566.yaml "
" NGFC 1047150 48744644 results/netclamp/network_clamp.optimize.NGFC_1047150_20210204_025430_48744644.yaml "
" NGFC 1047150 50965365 results/netclamp/network_clamp.optimize.NGFC_1047150_20210204_025430_50965365.yaml "
" NGFC 1048400 35297351 results/netclamp/network_clamp.optimize.NGFC_1048400_20210204_025430_35297351.yaml "
" NGFC 1048400 35472841 results/netclamp/network_clamp.optimize.NGFC_1048400_20210204_025430_35472841.yaml "
" NGFC 1048400 38650070 results/netclamp/network_clamp.optimize.NGFC_1048400_20210204_025430_38650070.yaml "
" NGFC 1048400 62046131 results/netclamp/network_clamp.optimize.NGFC_1048400_20210204_025430_62046131.yaml "
" NGFC 1048400 80347840 results/netclamp/network_clamp.optimize.NGFC_1048400_20210204_025430_80347840.yaml "
" NGFC 1049649 10249569 results/netclamp/network_clamp.optimize.NGFC_1049649_20210204_025429_10249569.yaml "
" NGFC 1049649 89481705 results/netclamp/network_clamp.optimize.NGFC_1049649_20210204_025429_89481705.yaml "
" NGFC 1049649 26628153 results/netclamp/network_clamp.optimize.NGFC_1049649_20210204_025430_26628153.yaml "
" NGFC 1049649 77179804 results/netclamp/network_clamp.optimize.NGFC_1049649_20210204_025430_77179804.yaml "
" NGFC 1049649 99082330 results/netclamp/network_clamp.optimize.NGFC_1049649_20210204_025430_99082330.yaml "
" IS 1049650 4259860 results/netclamp/network_clamp.optimize.IS_1049650_20210204_025429_04259860.yaml "
" IS 1049650 11745958 results/netclamp/network_clamp.optimize.IS_1049650_20210204_025429_11745958.yaml "
" IS 1049650 49627038 results/netclamp/network_clamp.optimize.IS_1049650_20210204_025430_49627038.yaml "
" IS 1049650 75940072 results/netclamp/network_clamp.optimize.IS_1049650_20210204_025430_75940072.yaml "
" IS 1049650 84013649 results/netclamp/network_clamp.optimize.IS_1049650_20210204_025430_84013649.yaml "
" IS 1050400 10084233 results/netclamp/network_clamp.optimize.IS_1050400_20210204_025429_10084233.yaml "
" IS 1050400 93591428 results/netclamp/network_clamp.optimize.IS_1050400_20210204_025429_93591428.yaml "
" IS 1050400 7843435 results/netclamp/network_clamp.optimize.IS_1050400_20210204_025430_07843435.yaml "
" IS 1050400 63796673 results/netclamp/network_clamp.optimize.IS_1050400_20210204_025430_63796673.yaml "
" IS 1050400 69320701 results/netclamp/network_clamp.optimize.IS_1050400_20210204_025430_69320701.yaml "
" IS 1051150 49587749 results/netclamp/network_clamp.optimize.IS_1051150_20210204_025428_49587749.yaml "
" IS 1051150 1339500 results/netclamp/network_clamp.optimize.IS_1051150_20210204_025429_01339500.yaml "
" IS 1051150 21032749 results/netclamp/network_clamp.optimize.IS_1051150_20210204_025429_21032749.yaml "
" IS 1051150 22725943 results/netclamp/network_clamp.optimize.IS_1051150_20210204_025429_22725943.yaml "
" IS 1051150 83916441 results/netclamp/network_clamp.optimize.IS_1051150_20210204_025429_83916441.yaml "
" IS 1051900 27654574 results/netclamp/network_clamp.optimize.IS_1051900_20210204_025428_27654574.yaml "
" IS 1051900 82185961 results/netclamp/network_clamp.optimize.IS_1051900_20210204_025428_82185961.yaml "
" IS 1051900 23672271 results/netclamp/network_clamp.optimize.IS_1051900_20210204_025430_23672271.yaml "
" IS 1051900 51871840 results/netclamp/network_clamp.optimize.IS_1051900_20210204_025430_51871840.yaml "
" IS 1051900 70119958 results/netclamp/network_clamp.optimize.IS_1051900_20210204_025430_70119958.yaml "
" IS 1052649 60814941 results/netclamp/network_clamp.optimize.IS_1052649_20210204_025429_60814941.yaml "
" IS 1052649 82004212 results/netclamp/network_clamp.optimize.IS_1052649_20210204_025429_82004212.yaml "
" IS 1052649 18680556 results/netclamp/network_clamp.optimize.IS_1052649_20210204_025430_18680556.yaml "
" IS 1052649 37549278 results/netclamp/network_clamp.optimize.IS_1052649_20210204_025430_37549278.yaml "
" IS 1052649 45707385 results/netclamp/network_clamp.optimize.IS_1052649_20210204_025430_45707385.yaml "
" MOPP 1052650 31571230 results/netclamp/network_clamp.optimize.MOPP_1052650_20210204_025429_31571230.yaml "
" MOPP 1052650 45373570 results/netclamp/network_clamp.optimize.MOPP_1052650_20210204_025429_45373570.yaml "
" MOPP 1052650 85763600 results/netclamp/network_clamp.optimize.MOPP_1052650_20210204_025429_85763600.yaml "
" MOPP 1052650 29079471 results/netclamp/network_clamp.optimize.MOPP_1052650_20210204_025430_29079471.yaml "
" MOPP 1052650 68839073 results/netclamp/network_clamp.optimize.MOPP_1052650_20210204_025430_68839073.yaml "
" MOPP 1053650 35281577 results/netclamp/network_clamp.optimize.MOPP_1053650_20210204_025430_35281577.yaml "
" MOPP 1053650 39888091 results/netclamp/network_clamp.optimize.MOPP_1053650_20210204_025430_39888091.yaml "
" MOPP 1053650 59550066 results/netclamp/network_clamp.optimize.MOPP_1053650_20210204_025430_59550066.yaml "
" MOPP 1053650 82093235 results/netclamp/network_clamp.optimize.MOPP_1053650_20210204_025430_82093235.yaml "
" MOPP 1053650 78038978 results/netclamp/network_clamp.optimize.MOPP_1053650_20210204_035942_78038978.yaml "
" MOPP 1054650 21247851 results/netclamp/network_clamp.optimize.MOPP_1054650_20210204_025430_21247851.yaml "
" MOPP 1054650 60567645 results/netclamp/network_clamp.optimize.MOPP_1054650_20210204_025430_60567645.yaml "
" MOPP 1054650 94967765 results/netclamp/network_clamp.optimize.MOPP_1054650_20210204_025430_94967765.yaml "
" MOPP 1054650 3611780 results/netclamp/network_clamp.optimize.MOPP_1054650_20210204_025431_03611780.yaml "
" MOPP 1054650 26628185 results/netclamp/network_clamp.optimize.MOPP_1054650_20210204_025431_26628185.yaml "
" MOPP 1055650 79924848 results/netclamp/network_clamp.optimize.MOPP_1055650_20210204_025429_79924848.yaml "
" MOPP 1055650 83145544 results/netclamp/network_clamp.optimize.MOPP_1055650_20210204_025429_83145544.yaml "
" MOPP 1055650 34097792 results/netclamp/network_clamp.optimize.MOPP_1055650_20210204_025431_34097792.yaml "
" MOPP 1055650 44866707 results/netclamp/network_clamp.optimize.MOPP_1055650_20210204_025431_44866707.yaml "
" MOPP 1055650 61810606 results/netclamp/network_clamp.optimize.MOPP_1055650_20210204_025431_61810606.yaml "
" MOPP 1056649 17666981 results/netclamp/network_clamp.optimize.MOPP_1056649_20210204_025429_17666981.yaml "
" MOPP 1056649 88486608 results/netclamp/network_clamp.optimize.MOPP_1056649_20210204_025429_88486608.yaml "
" MOPP 1056649 92808036 results/netclamp/network_clamp.optimize.MOPP_1056649_20210204_025429_92808036.yaml "
" MOPP 1056649 68347478 results/netclamp/network_clamp.optimize.MOPP_1056649_20210204_025430_68347478.yaml "
" MOPP 1056649 73504121 results/netclamp/network_clamp.optimize.MOPP_1056649_20210204_025430_73504121.yaml "
)

pil=(
" AAC 1042913 291547 results/netclamp/network_clamp.optimize.AAC_1042913_20210204_025429_00291547.yaml "
" AAC 1043025 82956063 results/netclamp/network_clamp.optimize.AAC_1043025_20210204_025429_82956063.yaml "
" AAC 1043138 40133402 results/netclamp/network_clamp.optimize.AAC_1043138_20210204_025430_40133402.yaml "
" AAC 1043138 82470709 results/netclamp/network_clamp.optimize.AAC_1043138_20210204_025430_82470709.yaml "
" AAC 1043138 85264434 results/netclamp/network_clamp.optimize.AAC_1043138_20210204_025430_85264434.yaml "
" AAC 1043249 66598438 results/netclamp/network_clamp.optimize.AAC_1043249_20210204_025429_66598438.yaml "
" HCC 1043600 64290895 results/netclamp/network_clamp.optimize.HCC_1043600_20210204_025430_64290895.yaml "
" HCC 1044300 92910319 results/netclamp/network_clamp.optimize.HCC_1044300_20210204_025429_92910319.yaml "
" HCC 1044300 94613541 results/netclamp/network_clamp.optimize.HCC_1044300_20210204_025429_94613541.yaml "
" HCC 1044300 97569363 results/netclamp/network_clamp.optimize.HCC_1044300_20210204_025430_97569363.yaml "
" HCC 1044649 94560988 results/netclamp/network_clamp.optimize.HCC_1044649_20210204_025429_94560988.yaml "
" NGFC 1044650 12740157 results/netclamp/network_clamp.optimize.NGFC_1044650_20210204_025430_12740157.yaml "
" NGFC 1045900 71039025 results/netclamp/network_clamp.optimize.NGFC_1045900_20210204_025429_71039025.yaml "
" NGFC 1045900 67428613 results/netclamp/network_clamp.optimize.NGFC_1045900_20210204_025431_67428613.yaml "
" NGFC 1047150 27400566 results/netclamp/network_clamp.optimize.NGFC_1047150_20210204_025430_27400566.yaml "
" NGFC 1047150 50965365 results/netclamp/network_clamp.optimize.NGFC_1047150_20210204_025430_50965365.yaml "
" NGFC 1049649 89481705 results/netclamp/network_clamp.optimize.NGFC_1049649_20210204_025429_89481705.yaml "
" NGFC 1049649 99082330 results/netclamp/network_clamp.optimize.NGFC_1049649_20210204_025430_99082330.yaml "
" IS 1049650 11745958 results/netclamp/network_clamp.optimize.IS_1049650_20210204_025429_11745958.yaml "
" IS 1051150 22725943 results/netclamp/network_clamp.optimize.IS_1051150_20210204_025429_22725943.yaml "
" IS 1051900 82185961 results/netclamp/network_clamp.optimize.IS_1051900_20210204_025428_82185961.yaml "
" IS 1052649 82004212 results/netclamp/network_clamp.optimize.IS_1052649_20210204_025429_82004212.yaml "
" IS 1052649 18680556 results/netclamp/network_clamp.optimize.IS_1052649_20210204_025430_18680556.yaml "
" MOPP 1052650 85763600 results/netclamp/network_clamp.optimize.MOPP_1052650_20210204_025429_85763600.yaml "
" MOPP 1053650 82093235 results/netclamp/network_clamp.optimize.MOPP_1053650_20210204_025430_82093235.yaml "
" MOPP 1054650 21247851 results/netclamp/network_clamp.optimize.MOPP_1054650_20210204_025430_21247851.yaml "
" MOPP 1054650 60567645 results/netclamp/network_clamp.optimize.MOPP_1054650_20210204_025430_60567645.yaml "
" MOPP 1054650 3611780 results/netclamp/network_clamp.optimize.MOPP_1054650_20210204_025431_03611780.yaml "
" MOPP 1054650 26628185 results/netclamp/network_clamp.optimize.MOPP_1054650_20210204_025431_26628185.yaml "
" MOPP 1055650 83145544 results/netclamp/network_clamp.optimize.MOPP_1055650_20210204_025429_83145544.yaml "
)

N_cores=1

IFS='
'
counter=0
for f in ${pil[@]}
do

set -- "$f" 
IFS=" " ; declare -a tempvar=($*) 


#ibrun -n 56 -o  0 task_affinity ./mycode.exe input1 &   # 56 tasks; offset by  0 entries in hostfile.
#ibrun -n 56 -o 56 task_affinity ./mycode.exe input2 &   # 56 tasks; offset by 56 entries in hostfile.
#wait                                                    # Required; else script will exit immediately.

#pop=${tempvar[0]:1:-1}
pop=${tempvar[0]}
gid=${tempvar[1]}
seed=${tempvar[2]}
yaml=${tempvar[3]}

ibrun -n $N_cores -o $counter python3 network_clamp.py go -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
    --template-paths templates \
    -p $pop -g $gid -t 9500 --dt 0.001 \
    --dataset-prefix /scratch1/03320/iraikov/striped/dentate \
    --config-prefix config \
    --input-features-path /scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --arena-id A --trajectory-id Diag \
    --results-path results/netclamp \
    --opt-seed $seed \
    --params-path $yaml &



##ibrun -n 8 python3  network_clamp.py optimize -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
#ibrun -n $N_cores -o $((counter * 56))  python3  network_clamp.py optimize -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
#    --template-paths templates \
#    -p $pop -g $gid -t 9500 --dt 0.001 \
#    --dataset-prefix /scratch1/03320/iraikov/striped/dentate \
#    --config-prefix config \
#    --input-features-path /scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
#    --input-features-namespaces 'Place Selectivity' \
#    --input-features-namespaces 'Grid Selectivity' \
#    --input-features-namespaces 'Constant Selectivity' \
#    --arena-id A --trajectory-id Diag \
#    --results-path results/netclamp \
#    --param-config-name "Weight exc inh microcircuit" \
#    --opt-seed $seed \
#    --opt-iter 400 rate & 


#    --results-file network_clamp.optimize.$pop\_$gid\_$(date +'%Y%m%d_%H%M%S')\_$seed.h5 \

counter=$((counter + 1))

done
wait
