graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    high 100
    low 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "max_cpu"
    originator "cpu"
    owner "node"
    type "extrema"
  ]
  link_attrs_setting [
    distribution "uniform"
    dtype "int"
    generative 1
    high 100
    low 50
    name "bw"
    owner "link"
    type "resource"
  ]
  link_attrs_setting [
    name "max_bw"
    originator "bw"
    owner "link"
    type "extrema"
  ]
  save_dir "dataset/p_net"
  topology [
    type "waxman"
    wm_alpha 0.5
    wm_beta 0.2
  ]
  file_name "p_net.gml"
  num_nodes 100
  type "waxman"
  wm_alpha 0.5
  wm_beta 0.2
  node [
    id 0
    label "0"
    pos 0.6268600538067624
    pos 0.7098160198040405
    cpu 84
    max_cpu 84
    gpu 76
    max_gpu 76
    ram 68
    max_ram 68
  ]
  node [
    id 1
    label "1"
    pos 0.472218109697886
    pos 0.7623817519525667
    cpu 86
    max_cpu 86
    gpu 53
    max_gpu 53
    ram 60
    max_ram 60
  ]
  node [
    id 2
    label "2"
    pos 0.2457904323715404
    pos 0.5120647101978727
    cpu 58
    max_cpu 58
    gpu 55
    max_gpu 55
    ram 72
    max_ram 72
  ]
  node [
    id 3
    label "3"
    pos 0.883915840982235
    pos 0.31069957402813875
    cpu 60
    max_cpu 60
    gpu 89
    max_gpu 89
    ram 78
    max_ram 78
  ]
  node [
    id 4
    label "4"
    pos 0.3078430203619492
    pos 0.5358500594314206
    cpu 63
    max_cpu 63
    gpu 66
    max_gpu 66
    ram 53
    max_ram 53
  ]
  node [
    id 5
    label "5"
    pos 0.32207997952121514
    pos 0.8863074771159885
    cpu 74
    max_cpu 74
    gpu 82
    max_gpu 82
    ram 54
    max_ram 54
  ]
  node [
    id 6
    label "6"
    pos 0.8676311522947634
    pos 0.9876929504939831
    cpu 78
    max_cpu 78
    gpu 91
    max_gpu 91
    ram 68
    max_ram 68
  ]
  node [
    id 7
    label "7"
    pos 0.540644070572585
    pos 0.5493780620912452
    cpu 100
    max_cpu 100
    gpu 57
    max_gpu 57
    ram 57
    max_ram 57
  ]
  node [
    id 8
    label "8"
    pos 0.5905582878928368
    pos 0.25154502731301864
    cpu 77
    max_cpu 77
    gpu 62
    max_gpu 62
    ram 55
    max_ram 55
  ]
  node [
    id 9
    label "9"
    pos 0.3403840345418758
    pos 0.7520406157268971
    cpu 61
    max_cpu 61
    gpu 79
    max_gpu 79
    ram 53
    max_ram 53
  ]
  node [
    id 10
    label "10"
    pos 0.6483913190793568
    pos 0.2986138316800897
    cpu 51
    max_cpu 51
    gpu 82
    max_gpu 82
    ram 90
    max_ram 90
  ]
  node [
    id 11
    label "11"
    pos 0.5773032256161507
    pos 0.041684053692682244
    cpu 61
    max_cpu 61
    gpu 67
    max_gpu 67
    ram 91
    max_ram 91
  ]
  node [
    id 12
    label "12"
    pos 0.3830167705556181
    pos 0.7455162657991262
    cpu 69
    max_cpu 69
    gpu 54
    max_gpu 54
    ram 72
    max_ram 72
  ]
  node [
    id 13
    label "13"
    pos 0.13914309576003936
    pos 0.0009991789869673307
    cpu 60
    max_cpu 60
    gpu 54
    max_gpu 54
    ram 79
    max_ram 79
  ]
  node [
    id 14
    label "14"
    pos 0.16017606563722553
    pos 0.9206665048213831
    cpu 61
    max_cpu 61
    gpu 64
    max_gpu 64
    ram 75
    max_ram 75
  ]
  node [
    id 15
    label "15"
    pos 0.2838607849412639
    pos 0.7054312207546886
    cpu 76
    max_cpu 76
    gpu 63
    max_gpu 63
    ram 60
    max_ram 60
  ]
  node [
    id 16
    label "16"
    pos 0.5410376780182408
    pos 0.2591122981604411
    cpu 52
    max_cpu 52
    gpu 93
    max_gpu 93
    ram 81
    max_ram 81
  ]
  node [
    id 17
    label "17"
    pos 0.1620204357567171
    pos 0.3641730600776728
    cpu 98
    max_cpu 98
    gpu 82
    max_gpu 82
    ram 90
    max_ram 90
  ]
  node [
    id 18
    label "18"
    pos 0.44388864341531187
    pos 0.29229874682088797
    cpu 78
    max_cpu 78
    gpu 81
    max_gpu 81
    ram 93
    max_ram 93
  ]
  node [
    id 19
    label "19"
    pos 0.8732652587892799
    pos 0.9079136706664074
    cpu 55
    max_cpu 55
    gpu 69
    max_gpu 69
    ram 86
    max_ram 86
  ]
  node [
    id 20
    label "20"
    pos 0.28826232230054716
    pos 0.2568348490101121
    cpu 55
    max_cpu 55
    gpu 97
    max_gpu 97
    ram 95
    max_ram 95
  ]
  node [
    id 21
    label "21"
    pos 0.8986733413250424
    pos 0.24033465889202466
    cpu 50
    max_cpu 50
    gpu 55
    max_gpu 55
    ram 91
    max_ram 91
  ]
  node [
    id 22
    label "22"
    pos 0.618539423036263
    pos 0.33965730807222605
    cpu 81
    max_cpu 81
    gpu 83
    max_gpu 83
    ram 67
    max_ram 67
  ]
  node [
    id 23
    label "23"
    pos 0.9469878824980217
    pos 0.11994278870781139
    cpu 81
    max_cpu 81
    gpu 84
    max_gpu 84
    ram 57
    max_ram 57
  ]
  node [
    id 24
    label "24"
    pos 0.8104325671604419
    pos 0.9810251812358353
    cpu 69
    max_cpu 69
    gpu 88
    max_gpu 88
    ram 67
    max_ram 67
  ]
  node [
    id 25
    label "25"
    pos 0.7407003923372613
    pos 0.6069417089077261
    cpu 61
    max_cpu 61
    gpu 88
    max_gpu 88
    ram 55
    max_ram 55
  ]
  node [
    id 26
    label "26"
    pos 0.05949861964918168
    pos 0.3988627192390989
    cpu 70
    max_cpu 70
    gpu 73
    max_gpu 73
    ram 87
    max_ram 87
  ]
  node [
    id 27
    label "27"
    pos 0.5115476044043714
    pos 0.518011477668278
    cpu 76
    max_cpu 76
    gpu 68
    max_gpu 68
    ram 67
    max_ram 67
  ]
  node [
    id 28
    label "28"
    pos 0.6908072755688102
    pos 0.4540138231817197
    cpu 100
    max_cpu 100
    gpu 55
    max_gpu 55
    ram 59
    max_ram 59
  ]
  node [
    id 29
    label "29"
    pos 0.5873823897642683
    pos 0.19706197012976545
    cpu 58
    max_cpu 58
    gpu 59
    max_gpu 59
    ram 87
    max_ram 87
  ]
  node [
    id 30
    label "30"
    pos 0.9443285853616543
    pos 0.19484765651047875
    cpu 65
    max_cpu 65
    gpu 94
    max_gpu 94
    ram 65
    max_ram 65
  ]
  node [
    id 31
    label "31"
    pos 0.9845736696816485
    pos 0.4193013071923032
    cpu 96
    max_cpu 96
    gpu 76
    max_gpu 76
    ram 92
    max_ram 92
  ]
  node [
    id 32
    label "32"
    pos 0.45816743878125465
    pos 0.9525128844337839
    cpu 60
    max_cpu 60
    gpu 77
    max_gpu 77
    ram 100
    max_ram 100
  ]
  node [
    id 33
    label "33"
    pos 0.855894664905305
    pos 0.3007864035061628
    cpu 66
    max_cpu 66
    gpu 70
    max_gpu 70
    ram 67
    max_ram 67
  ]
  node [
    id 34
    label "34"
    pos 0.0873209717880945
    pos 0.29159768892232085
    cpu 78
    max_cpu 78
    gpu 69
    max_gpu 69
    ram 69
    max_ram 69
  ]
  node [
    id 35
    label "35"
    pos 0.8752368592882798
    pos 0.610539995355712
    cpu 77
    max_cpu 77
    gpu 72
    max_gpu 72
    ram 54
    max_ram 54
  ]
  node [
    id 36
    label "36"
    pos 0.6933581299201721
    pos 0.7991415728526647
    cpu 85
    max_cpu 85
    gpu 67
    max_gpu 67
    ram 63
    max_ram 63
  ]
  node [
    id 37
    label "37"
    pos 0.11636980016150822
    pos 0.21712707336154669
    cpu 61
    max_cpu 61
    gpu 87
    max_gpu 87
    ram 58
    max_ram 58
  ]
  node [
    id 38
    label "38"
    pos 0.15636915186198985
    pos 0.07881057634492716
    cpu 97
    max_cpu 97
    gpu 76
    max_gpu 76
    ram 83
    max_ram 83
  ]
  node [
    id 39
    label "39"
    pos 0.36831518085737325
    pos 0.7205042901168887
    cpu 68
    max_cpu 68
    gpu 80
    max_gpu 80
    ram 92
    max_ram 92
  ]
  node [
    id 40
    label "40"
    pos 0.18350002514309005
    pos 0.38276288109047607
    cpu 53
    max_cpu 53
    gpu 91
    max_gpu 91
    ram 69
    max_ram 69
  ]
  node [
    id 41
    label "41"
    pos 0.37498195809891177
    pos 0.3054920160741915
    cpu 88
    max_cpu 88
    gpu 92
    max_gpu 92
    ram 76
    max_ram 76
  ]
  node [
    id 42
    label "42"
    pos 0.26063303883084077
    pos 0.8910066132503863
    cpu 69
    max_cpu 69
    gpu 54
    max_gpu 54
    ram 91
    max_ram 91
  ]
  node [
    id 43
    label "43"
    pos 0.07426178983012144
    pos 0.1477502453718632
    cpu 81
    max_cpu 81
    gpu 76
    max_gpu 76
    ram 52
    max_ram 52
  ]
  node [
    id 44
    label "44"
    pos 0.48591377302298244
    pos 0.9648415189514911
    cpu 96
    max_cpu 96
    gpu 59
    max_gpu 59
    ram 62
    max_ram 62
  ]
  node [
    id 45
    label "45"
    pos 0.760795154146904
    pos 0.7166524452413132
    cpu 92
    max_cpu 92
    gpu 100
    max_gpu 100
    ram 77
    max_ram 77
  ]
  node [
    id 46
    label "46"
    pos 0.8110073990086403
    pos 0.7008121136547419
    cpu 56
    max_cpu 56
    gpu 57
    max_gpu 57
    ram 58
    max_ram 58
  ]
  node [
    id 47
    label "47"
    pos 0.7713851811192931
    pos 0.040275610441808385
    cpu 86
    max_cpu 86
    gpu 74
    max_gpu 74
    ram 79
    max_ram 79
  ]
  node [
    id 48
    label "48"
    pos 0.7486149701390689
    pos 0.2161271128226845
    cpu 79
    max_cpu 79
    gpu 70
    max_gpu 70
    ram 62
    max_ram 62
  ]
  node [
    id 49
    label "49"
    pos 0.23541251490046344
    pos 0.26507147670583475
    cpu 53
    max_cpu 53
    gpu 60
    max_gpu 60
    ram 90
    max_ram 90
  ]
  node [
    id 50
    label "50"
    pos 0.8647334954415141
    pos 0.6294179721904186
    cpu 79
    max_cpu 79
    gpu 87
    max_gpu 87
    ram 65
    max_ram 65
  ]
  node [
    id 51
    label "51"
    pos 0.3143144325546664
    pos 0.5944349253119335
    cpu 84
    max_cpu 84
    gpu 76
    max_gpu 76
    ram 88
    max_ram 88
  ]
  node [
    id 52
    label "52"
    pos 0.8433421564517695
    pos 0.6004329885118707
    cpu 75
    max_cpu 75
    gpu 57
    max_gpu 57
    ram 96
    max_ram 96
  ]
  node [
    id 53
    label "53"
    pos 0.4091384240768354
    pos 0.936735808046114
    cpu 72
    max_cpu 72
    gpu 79
    max_gpu 79
    ram 98
    max_ram 98
  ]
  node [
    id 54
    label "54"
    pos 0.7077690509337657
    pos 0.01085824973266214
    cpu 55
    max_cpu 55
    gpu 71
    max_gpu 71
    ram 91
    max_ram 91
  ]
  node [
    id 55
    label "55"
    pos 0.6493393911120028
    pos 0.2598089547984277
    cpu 88
    max_cpu 88
    gpu 85
    max_gpu 85
    ram 65
    max_ram 65
  ]
  node [
    id 56
    label "56"
    pos 0.6165703644071431
    pos 0.49119200755215686
    cpu 78
    max_cpu 78
    gpu 84
    max_gpu 84
    ram 84
    max_ram 84
  ]
  node [
    id 57
    label "57"
    pos 0.6440093840964123
    pos 0.27312000973447603
    cpu 94
    max_cpu 94
    gpu 87
    max_gpu 87
    ram 83
    max_ram 83
  ]
  node [
    id 58
    label "58"
    pos 0.28910281952516304
    pos 0.7623837009655622
    cpu 53
    max_cpu 53
    gpu 87
    max_gpu 87
    ram 91
    max_ram 91
  ]
  node [
    id 59
    label "59"
    pos 0.5286185051246081
    pos 0.5358178020939711
    cpu 78
    max_cpu 78
    gpu 51
    max_gpu 51
    ram 63
    max_ram 63
  ]
  node [
    id 60
    label "60"
    pos 0.6594055137800715
    pos 0.46484605032037063
    cpu 70
    max_cpu 70
    gpu 98
    max_gpu 98
    ram 79
    max_ram 79
  ]
  node [
    id 61
    label "61"
    pos 0.9388811563247641
    pos 0.6300558272105845
    cpu 67
    max_cpu 67
    gpu 60
    max_gpu 60
    ram 100
    max_ram 100
  ]
  node [
    id 62
    label "62"
    pos 0.10885128066218641
    pos 0.7538481761323483
    cpu 85
    max_cpu 85
    gpu 95
    max_gpu 95
    ram 96
    max_ram 96
  ]
  node [
    id 63
    label "63"
    pos 0.9206238995956619
    pos 0.887222160384117
    cpu 59
    max_cpu 59
    gpu 89
    max_gpu 89
    ram 56
    max_ram 56
  ]
  node [
    id 64
    label "64"
    pos 0.7314937324216402
    pos 0.675937376170674
    cpu 58
    max_cpu 58
    gpu 100
    max_gpu 100
    ram 88
    max_ram 88
  ]
  node [
    id 65
    label "65"
    pos 0.3863457963370672
    pos 0.39478822792979473
    cpu 88
    max_cpu 88
    gpu 58
    max_gpu 58
    ram 78
    max_ram 78
  ]
  node [
    id 66
    label "66"
    pos 0.7040000863918565
    pos 0.18139350690445533
    cpu 77
    max_cpu 77
    gpu 78
    max_gpu 78
    ram 82
    max_ram 82
  ]
  node [
    id 67
    label "67"
    pos 0.9443279567957064
    pos 0.24870598103754138
    cpu 78
    max_cpu 78
    gpu 99
    max_gpu 99
    ram 88
    max_ram 88
  ]
  node [
    id 68
    label "68"
    pos 0.541505153536048
    pos 0.24332956977982456
    cpu 73
    max_cpu 73
    gpu 70
    max_gpu 70
    ram 61
    max_ram 61
  ]
  node [
    id 69
    label "69"
    pos 0.33790874590143416
    pos 0.5050371121402387
    cpu 69
    max_cpu 69
    gpu 95
    max_gpu 95
    ram 93
    max_ram 93
  ]
  node [
    id 70
    label "70"
    pos 0.6090651987574806
    pos 0.6063140034680162
    cpu 97
    max_cpu 97
    gpu 77
    max_gpu 77
    ram 92
    max_ram 92
  ]
  node [
    id 71
    label "71"
    pos 0.931096198831702
    pos 0.000272214533331927
    cpu 93
    max_cpu 93
    gpu 95
    max_gpu 95
    ram 68
    max_ram 68
  ]
  node [
    id 72
    label "72"
    pos 0.9606830640980398
    pos 0.80615815462669
    cpu 87
    max_cpu 87
    gpu 56
    max_gpu 56
    ram 89
    max_ram 89
  ]
  node [
    id 73
    label "73"
    pos 0.05175000211776615
    pos 0.6987611676311768
    cpu 68
    max_cpu 68
    gpu 87
    max_gpu 87
    ram 68
    max_ram 68
  ]
  node [
    id 74
    label "74"
    pos 0.0984848371111503
    pos 0.45923378704440754
    cpu 98
    max_cpu 98
    gpu 91
    max_gpu 91
    ram 54
    max_ram 54
  ]
  node [
    id 75
    label "75"
    pos 0.489009412872077
    pos 0.3844263921694824
    cpu 74
    max_cpu 74
    gpu 53
    max_gpu 53
    ram 68
    max_ram 68
  ]
  node [
    id 76
    label "76"
    pos 0.44152760842798133
    pos 0.32013318241437316
    cpu 59
    max_cpu 59
    gpu 76
    max_gpu 76
    ram 58
    max_ram 58
  ]
  node [
    id 77
    label "77"
    pos 0.7677557109544005
    pos 0.2047875940147248
    cpu 79
    max_cpu 79
    gpu 93
    max_gpu 93
    ram 52
    max_ram 52
  ]
  node [
    id 78
    label "78"
    pos 0.482394144291027
    pos 0.045399130369401686
    cpu 73
    max_cpu 73
    gpu 61
    max_gpu 61
    ram 50
    max_ram 50
  ]
  node [
    id 79
    label "79"
    pos 0.8391313080131036
    pos 0.7862014462381003
    cpu 58
    max_cpu 58
    gpu 65
    max_gpu 65
    ram 69
    max_ram 69
  ]
  node [
    id 80
    label "80"
    pos 0.49051503365290294
    pos 0.3840381668752353
    cpu 60
    max_cpu 60
    gpu 52
    max_gpu 52
    ram 61
    max_ram 61
  ]
  node [
    id 81
    label "81"
    pos 0.030321412272943937
    pos 0.081119887402469
    cpu 60
    max_cpu 60
    gpu 62
    max_gpu 62
    ram 84
    max_ram 84
  ]
  node [
    id 82
    label "82"
    pos 0.6236944611614518
    pos 0.9012742271177318
    cpu 97
    max_cpu 97
    gpu 82
    max_gpu 82
    ram 74
    max_ram 74
  ]
  node [
    id 83
    label "83"
    pos 0.05327006311209881
    pos 0.7834307006456989
    cpu 93
    max_cpu 93
    gpu 70
    max_gpu 70
    ram 64
    max_ram 64
  ]
  node [
    id 84
    label "84"
    pos 0.501966286178314
    pos 0.03346241108773229
    cpu 81
    max_cpu 81
    gpu 62
    max_gpu 62
    ram 55
    max_ram 55
  ]
  node [
    id 85
    label "85"
    pos 0.8942163404831253
    pos 0.016884923714836964
    cpu 60
    max_cpu 60
    gpu 69
    max_gpu 69
    ram 80
    max_ram 80
  ]
  node [
    id 86
    label "86"
    pos 0.5051109174911756
    pos 0.36543402419950355
    cpu 91
    max_cpu 91
    gpu 87
    max_gpu 87
    ram 100
    max_ram 100
  ]
  node [
    id 87
    label "87"
    pos 0.1960176267713083
    pos 0.725760660980179
    cpu 81
    max_cpu 81
    gpu 67
    max_gpu 67
    ram 73
    max_ram 73
  ]
  node [
    id 88
    label "88"
    pos 0.20383983856272514
    pos 0.40690663269707017
    cpu 91
    max_cpu 91
    gpu 62
    max_gpu 62
    ram 75
    max_ram 75
  ]
  node [
    id 89
    label "89"
    pos 0.08804078569126406
    pos 0.615169242386062
    cpu 72
    max_cpu 72
    gpu 61
    max_gpu 61
    ram 54
    max_ram 54
  ]
  node [
    id 90
    label "90"
    pos 0.9758092250335976
    pos 0.3669645679256279
    cpu 96
    max_cpu 96
    gpu 85
    max_gpu 85
    ram 73
    max_ram 73
  ]
  node [
    id 91
    label "91"
    pos 0.13502404320129824
    pos 0.9727950146191573
    cpu 68
    max_cpu 68
    gpu 70
    max_gpu 70
    ram 89
    max_ram 89
  ]
  node [
    id 92
    label "92"
    pos 0.7653650989191546
    pos 0.15160773086610102
    cpu 100
    max_cpu 100
    gpu 85
    max_gpu 85
    ram 54
    max_ram 54
  ]
  node [
    id 93
    label "93"
    pos 0.6616953486330137
    pos 0.8583324453222967
    cpu 57
    max_cpu 57
    gpu 98
    max_gpu 98
    ram 51
    max_ram 51
  ]
  node [
    id 94
    label "94"
    pos 0.7784478096189543
    pos 0.7189301277481611
    cpu 65
    max_cpu 65
    gpu 100
    max_gpu 100
    ram 91
    max_ram 91
  ]
  node [
    id 95
    label "95"
    pos 0.8740052019518417
    pos 0.7877287998667755
    cpu 54
    max_cpu 54
    gpu 74
    max_gpu 74
    ram 66
    max_ram 66
  ]
  node [
    id 96
    label "96"
    pos 0.5576586933107668
    pos 0.5337188077285331
    cpu 80
    max_cpu 80
    gpu 69
    max_gpu 69
    ram 75
    max_ram 75
  ]
  node [
    id 97
    label "97"
    pos 0.7473774300440983
    pos 0.2560476077112286
    cpu 50
    max_cpu 50
    gpu 63
    max_gpu 63
    ram 65
    max_ram 65
  ]
  node [
    id 98
    label "98"
    pos 0.4321443368867517
    pos 0.7106555998791511
    cpu 84
    max_cpu 84
    gpu 63
    max_gpu 63
    ram 79
    max_ram 79
  ]
  node [
    id 99
    label "99"
    pos 0.5114861722500027
    pos 0.022435647158953742
    cpu 77
    max_cpu 77
    gpu 89
    max_gpu 89
    ram 50
    max_ram 50
  ]
  edge [
    source 0
    target 4
    bw 51
    max_bw 51
  ]
  edge [
    source 0
    target 6
    bw 68
    max_bw 68
  ]
  edge [
    source 0
    target 7
    bw 82
    max_bw 82
  ]
  edge [
    source 0
    target 9
    bw 55
    max_bw 55
  ]
  edge [
    source 0
    target 17
    bw 76
    max_bw 76
  ]
  edge [
    source 0
    target 35
    bw 86
    max_bw 86
  ]
  edge [
    source 0
    target 36
    bw 53
    max_bw 53
  ]
  edge [
    source 0
    target 46
    bw 51
    max_bw 51
  ]
  edge [
    source 0
    target 65
    bw 52
    max_bw 52
  ]
  edge [
    source 0
    target 82
    bw 55
    max_bw 55
  ]
  edge [
    source 0
    target 86
    bw 51
    max_bw 51
  ]
  edge [
    source 0
    target 93
    bw 69
    max_bw 69
  ]
  edge [
    source 0
    target 95
    bw 51
    max_bw 51
  ]
  edge [
    source 1
    target 4
    bw 87
    max_bw 87
  ]
  edge [
    source 1
    target 9
    bw 58
    max_bw 58
  ]
  edge [
    source 1
    target 12
    bw 67
    max_bw 67
  ]
  edge [
    source 1
    target 25
    bw 54
    max_bw 54
  ]
  edge [
    source 1
    target 26
    bw 96
    max_bw 96
  ]
  edge [
    source 1
    target 66
    bw 90
    max_bw 90
  ]
  edge [
    source 1
    target 70
    bw 79
    max_bw 79
  ]
  edge [
    source 1
    target 76
    bw 68
    max_bw 68
  ]
  edge [
    source 1
    target 79
    bw 60
    max_bw 60
  ]
  edge [
    source 1
    target 82
    bw 62
    max_bw 62
  ]
  edge [
    source 1
    target 96
    bw 50
    max_bw 50
  ]
  edge [
    source 2
    target 4
    bw 77
    max_bw 77
  ]
  edge [
    source 2
    target 15
    bw 62
    max_bw 62
  ]
  edge [
    source 2
    target 26
    bw 60
    max_bw 60
  ]
  edge [
    source 2
    target 32
    bw 86
    max_bw 86
  ]
  edge [
    source 2
    target 56
    bw 66
    max_bw 66
  ]
  edge [
    source 2
    target 62
    bw 53
    max_bw 53
  ]
  edge [
    source 2
    target 74
    bw 84
    max_bw 84
  ]
  edge [
    source 2
    target 75
    bw 74
    max_bw 74
  ]
  edge [
    source 2
    target 80
    bw 90
    max_bw 90
  ]
  edge [
    source 2
    target 81
    bw 50
    max_bw 50
  ]
  edge [
    source 2
    target 93
    bw 75
    max_bw 75
  ]
  edge [
    source 3
    target 13
    bw 95
    max_bw 95
  ]
  edge [
    source 3
    target 23
    bw 67
    max_bw 67
  ]
  edge [
    source 3
    target 33
    bw 59
    max_bw 59
  ]
  edge [
    source 3
    target 39
    bw 55
    max_bw 55
  ]
  edge [
    source 3
    target 45
    bw 57
    max_bw 57
  ]
  edge [
    source 3
    target 48
    bw 91
    max_bw 91
  ]
  edge [
    source 3
    target 50
    bw 70
    max_bw 70
  ]
  edge [
    source 3
    target 67
    bw 87
    max_bw 87
  ]
  edge [
    source 3
    target 70
    bw 92
    max_bw 92
  ]
  edge [
    source 3
    target 71
    bw 90
    max_bw 90
  ]
  edge [
    source 3
    target 90
    bw 75
    max_bw 75
  ]
  edge [
    source 4
    target 5
    bw 89
    max_bw 89
  ]
  edge [
    source 4
    target 7
    bw 97
    max_bw 97
  ]
  edge [
    source 4
    target 9
    bw 59
    max_bw 59
  ]
  edge [
    source 4
    target 18
    bw 86
    max_bw 86
  ]
  edge [
    source 4
    target 25
    bw 91
    max_bw 91
  ]
  edge [
    source 4
    target 26
    bw 93
    max_bw 93
  ]
  edge [
    source 4
    target 41
    bw 78
    max_bw 78
  ]
  edge [
    source 4
    target 51
    bw 76
    max_bw 76
  ]
  edge [
    source 4
    target 59
    bw 97
    max_bw 97
  ]
  edge [
    source 4
    target 62
    bw 76
    max_bw 76
  ]
  edge [
    source 4
    target 68
    bw 77
    max_bw 77
  ]
  edge [
    source 4
    target 86
    bw 64
    max_bw 64
  ]
  edge [
    source 4
    target 93
    bw 67
    max_bw 67
  ]
  edge [
    source 4
    target 99
    bw 63
    max_bw 63
  ]
  edge [
    source 5
    target 6
    bw 69
    max_bw 69
  ]
  edge [
    source 5
    target 9
    bw 96
    max_bw 96
  ]
  edge [
    source 5
    target 10
    bw 75
    max_bw 75
  ]
  edge [
    source 5
    target 36
    bw 75
    max_bw 75
  ]
  edge [
    source 5
    target 42
    bw 55
    max_bw 55
  ]
  edge [
    source 5
    target 93
    bw 61
    max_bw 61
  ]
  edge [
    source 6
    target 25
    bw 82
    max_bw 82
  ]
  edge [
    source 6
    target 31
    bw 79
    max_bw 79
  ]
  edge [
    source 6
    target 32
    bw 64
    max_bw 64
  ]
  edge [
    source 6
    target 63
    bw 76
    max_bw 76
  ]
  edge [
    source 6
    target 67
    bw 62
    max_bw 62
  ]
  edge [
    source 6
    target 95
    bw 96
    max_bw 96
  ]
  edge [
    source 7
    target 9
    bw 50
    max_bw 50
  ]
  edge [
    source 7
    target 10
    bw 58
    max_bw 58
  ]
  edge [
    source 7
    target 11
    bw 100
    max_bw 100
  ]
  edge [
    source 7
    target 25
    bw 76
    max_bw 76
  ]
  edge [
    source 7
    target 42
    bw 64
    max_bw 64
  ]
  edge [
    source 7
    target 58
    bw 90
    max_bw 90
  ]
  edge [
    source 7
    target 59
    bw 96
    max_bw 96
  ]
  edge [
    source 7
    target 60
    bw 83
    max_bw 83
  ]
  edge [
    source 7
    target 68
    bw 87
    max_bw 87
  ]
  edge [
    source 7
    target 70
    bw 52
    max_bw 52
  ]
  edge [
    source 7
    target 75
    bw 65
    max_bw 65
  ]
  edge [
    source 7
    target 79
    bw 68
    max_bw 68
  ]
  edge [
    source 7
    target 93
    bw 63
    max_bw 63
  ]
  edge [
    source 8
    target 17
    bw 96
    max_bw 96
  ]
  edge [
    source 8
    target 24
    bw 99
    max_bw 99
  ]
  edge [
    source 8
    target 27
    bw 86
    max_bw 86
  ]
  edge [
    source 8
    target 28
    bw 96
    max_bw 96
  ]
  edge [
    source 8
    target 33
    bw 86
    max_bw 86
  ]
  edge [
    source 8
    target 39
    bw 76
    max_bw 76
  ]
  edge [
    source 8
    target 46
    bw 61
    max_bw 61
  ]
  edge [
    source 8
    target 66
    bw 99
    max_bw 99
  ]
  edge [
    source 8
    target 68
    bw 76
    max_bw 76
  ]
  edge [
    source 8
    target 75
    bw 73
    max_bw 73
  ]
  edge [
    source 8
    target 77
    bw 92
    max_bw 92
  ]
  edge [
    source 8
    target 84
    bw 70
    max_bw 70
  ]
  edge [
    source 8
    target 96
    bw 53
    max_bw 53
  ]
  edge [
    source 9
    target 39
    bw 82
    max_bw 82
  ]
  edge [
    source 9
    target 50
    bw 73
    max_bw 73
  ]
  edge [
    source 9
    target 93
    bw 63
    max_bw 63
  ]
  edge [
    source 10
    target 18
    bw 92
    max_bw 92
  ]
  edge [
    source 10
    target 21
    bw 87
    max_bw 87
  ]
  edge [
    source 10
    target 22
    bw 89
    max_bw 89
  ]
  edge [
    source 10
    target 29
    bw 78
    max_bw 78
  ]
  edge [
    source 10
    target 56
    bw 81
    max_bw 81
  ]
  edge [
    source 10
    target 60
    bw 63
    max_bw 63
  ]
  edge [
    source 10
    target 68
    bw 75
    max_bw 75
  ]
  edge [
    source 10
    target 69
    bw 75
    max_bw 75
  ]
  edge [
    source 11
    target 13
    bw 75
    max_bw 75
  ]
  edge [
    source 11
    target 38
    bw 80
    max_bw 80
  ]
  edge [
    source 11
    target 42
    bw 62
    max_bw 62
  ]
  edge [
    source 11
    target 78
    bw 95
    max_bw 95
  ]
  edge [
    source 11
    target 86
    bw 88
    max_bw 88
  ]
  edge [
    source 12
    target 34
    bw 55
    max_bw 55
  ]
  edge [
    source 12
    target 39
    bw 65
    max_bw 65
  ]
  edge [
    source 12
    target 68
    bw 87
    max_bw 87
  ]
  edge [
    source 12
    target 69
    bw 96
    max_bw 96
  ]
  edge [
    source 12
    target 70
    bw 78
    max_bw 78
  ]
  edge [
    source 12
    target 74
    bw 53
    max_bw 53
  ]
  edge [
    source 12
    target 78
    bw 82
    max_bw 82
  ]
  edge [
    source 12
    target 79
    bw 57
    max_bw 57
  ]
  edge [
    source 12
    target 96
    bw 97
    max_bw 97
  ]
  edge [
    source 13
    target 20
    bw 79
    max_bw 79
  ]
  edge [
    source 13
    target 41
    bw 84
    max_bw 84
  ]
  edge [
    source 13
    target 43
    bw 59
    max_bw 59
  ]
  edge [
    source 13
    target 62
    bw 78
    max_bw 78
  ]
  edge [
    source 13
    target 68
    bw 94
    max_bw 94
  ]
  edge [
    source 13
    target 84
    bw 79
    max_bw 79
  ]
  edge [
    source 13
    target 97
    bw 68
    max_bw 68
  ]
  edge [
    source 14
    target 18
    bw 56
    max_bw 56
  ]
  edge [
    source 14
    target 30
    bw 88
    max_bw 88
  ]
  edge [
    source 14
    target 36
    bw 56
    max_bw 56
  ]
  edge [
    source 14
    target 40
    bw 74
    max_bw 74
  ]
  edge [
    source 14
    target 53
    bw 64
    max_bw 64
  ]
  edge [
    source 14
    target 55
    bw 87
    max_bw 87
  ]
  edge [
    source 15
    target 20
    bw 58
    max_bw 58
  ]
  edge [
    source 15
    target 89
    bw 90
    max_bw 90
  ]
  edge [
    source 16
    target 18
    bw 83
    max_bw 83
  ]
  edge [
    source 16
    target 26
    bw 82
    max_bw 82
  ]
  edge [
    source 16
    target 30
    bw 99
    max_bw 99
  ]
  edge [
    source 16
    target 40
    bw 60
    max_bw 60
  ]
  edge [
    source 16
    target 41
    bw 59
    max_bw 59
  ]
  edge [
    source 16
    target 46
    bw 72
    max_bw 72
  ]
  edge [
    source 16
    target 65
    bw 57
    max_bw 57
  ]
  edge [
    source 16
    target 76
    bw 58
    max_bw 58
  ]
  edge [
    source 16
    target 78
    bw 94
    max_bw 94
  ]
  edge [
    source 17
    target 18
    bw 80
    max_bw 80
  ]
  edge [
    source 17
    target 21
    bw 53
    max_bw 53
  ]
  edge [
    source 17
    target 35
    bw 100
    max_bw 100
  ]
  edge [
    source 17
    target 41
    bw 51
    max_bw 51
  ]
  edge [
    source 17
    target 48
    bw 85
    max_bw 85
  ]
  edge [
    source 17
    target 56
    bw 52
    max_bw 52
  ]
  edge [
    source 17
    target 68
    bw 61
    max_bw 61
  ]
  edge [
    source 17
    target 74
    bw 95
    max_bw 95
  ]
  edge [
    source 17
    target 76
    bw 51
    max_bw 51
  ]
  edge [
    source 17
    target 78
    bw 51
    max_bw 51
  ]
  edge [
    source 17
    target 86
    bw 97
    max_bw 97
  ]
  edge [
    source 17
    target 88
    bw 89
    max_bw 89
  ]
  edge [
    source 17
    target 98
    bw 70
    max_bw 70
  ]
  edge [
    source 18
    target 22
    bw 68
    max_bw 68
  ]
  edge [
    source 18
    target 27
    bw 74
    max_bw 74
  ]
  edge [
    source 18
    target 30
    bw 70
    max_bw 70
  ]
  edge [
    source 18
    target 38
    bw 83
    max_bw 83
  ]
  edge [
    source 18
    target 40
    bw 78
    max_bw 78
  ]
  edge [
    source 18
    target 49
    bw 70
    max_bw 70
  ]
  edge [
    source 18
    target 55
    bw 95
    max_bw 95
  ]
  edge [
    source 18
    target 60
    bw 81
    max_bw 81
  ]
  edge [
    source 18
    target 65
    bw 80
    max_bw 80
  ]
  edge [
    source 18
    target 68
    bw 63
    max_bw 63
  ]
  edge [
    source 18
    target 97
    bw 55
    max_bw 55
  ]
  edge [
    source 19
    target 24
    bw 72
    max_bw 72
  ]
  edge [
    source 19
    target 36
    bw 89
    max_bw 89
  ]
  edge [
    source 19
    target 50
    bw 74
    max_bw 74
  ]
  edge [
    source 19
    target 64
    bw 85
    max_bw 85
  ]
  edge [
    source 19
    target 72
    bw 96
    max_bw 96
  ]
  edge [
    source 19
    target 79
    bw 72
    max_bw 72
  ]
  edge [
    source 19
    target 93
    bw 98
    max_bw 98
  ]
  edge [
    source 19
    target 95
    bw 92
    max_bw 92
  ]
  edge [
    source 20
    target 22
    bw 72
    max_bw 72
  ]
  edge [
    source 20
    target 38
    bw 64
    max_bw 64
  ]
  edge [
    source 20
    target 40
    bw 92
    max_bw 92
  ]
  edge [
    source 20
    target 41
    bw 81
    max_bw 81
  ]
  edge [
    source 20
    target 43
    bw 84
    max_bw 84
  ]
  edge [
    source 20
    target 49
    bw 75
    max_bw 75
  ]
  edge [
    source 20
    target 55
    bw 91
    max_bw 91
  ]
  edge [
    source 20
    target 58
    bw 74
    max_bw 74
  ]
  edge [
    source 20
    target 74
    bw 79
    max_bw 79
  ]
  edge [
    source 20
    target 85
    bw 72
    max_bw 72
  ]
  edge [
    source 20
    target 92
    bw 86
    max_bw 86
  ]
  edge [
    source 20
    target 98
    bw 98
    max_bw 98
  ]
  edge [
    source 21
    target 31
    bw 95
    max_bw 95
  ]
  edge [
    source 21
    target 55
    bw 71
    max_bw 71
  ]
  edge [
    source 21
    target 61
    bw 80
    max_bw 80
  ]
  edge [
    source 21
    target 68
    bw 74
    max_bw 74
  ]
  edge [
    source 21
    target 75
    bw 72
    max_bw 72
  ]
  edge [
    source 21
    target 85
    bw 88
    max_bw 88
  ]
  edge [
    source 21
    target 94
    bw 64
    max_bw 64
  ]
  edge [
    source 21
    target 97
    bw 91
    max_bw 91
  ]
  edge [
    source 22
    target 26
    bw 56
    max_bw 56
  ]
  edge [
    source 22
    target 38
    bw 66
    max_bw 66
  ]
  edge [
    source 22
    target 50
    bw 77
    max_bw 77
  ]
  edge [
    source 22
    target 54
    bw 90
    max_bw 90
  ]
  edge [
    source 22
    target 60
    bw 61
    max_bw 61
  ]
  edge [
    source 22
    target 63
    bw 60
    max_bw 60
  ]
  edge [
    source 22
    target 76
    bw 68
    max_bw 68
  ]
  edge [
    source 22
    target 80
    bw 95
    max_bw 95
  ]
  edge [
    source 22
    target 86
    bw 57
    max_bw 57
  ]
  edge [
    source 22
    target 94
    bw 56
    max_bw 56
  ]
  edge [
    source 23
    target 31
    bw 100
    max_bw 100
  ]
  edge [
    source 23
    target 66
    bw 68
    max_bw 68
  ]
  edge [
    source 23
    target 71
    bw 76
    max_bw 76
  ]
  edge [
    source 23
    target 72
    bw 78
    max_bw 78
  ]
  edge [
    source 23
    target 91
    bw 61
    max_bw 61
  ]
  edge [
    source 24
    target 25
    bw 69
    max_bw 69
  ]
  edge [
    source 24
    target 39
    bw 82
    max_bw 82
  ]
  edge [
    source 24
    target 63
    bw 60
    max_bw 60
  ]
  edge [
    source 24
    target 64
    bw 94
    max_bw 94
  ]
  edge [
    source 24
    target 93
    bw 72
    max_bw 72
  ]
  edge [
    source 24
    target 98
    bw 57
    max_bw 57
  ]
  edge [
    source 25
    target 27
    bw 87
    max_bw 87
  ]
  edge [
    source 25
    target 39
    bw 81
    max_bw 81
  ]
  edge [
    source 25
    target 60
    bw 65
    max_bw 65
  ]
  edge [
    source 25
    target 61
    bw 77
    max_bw 77
  ]
  edge [
    source 25
    target 93
    bw 77
    max_bw 77
  ]
  edge [
    source 25
    target 98
    bw 68
    max_bw 68
  ]
  edge [
    source 26
    target 47
    bw 55
    max_bw 55
  ]
  edge [
    source 26
    target 74
    bw 90
    max_bw 90
  ]
  edge [
    source 26
    target 77
    bw 89
    max_bw 89
  ]
  edge [
    source 26
    target 83
    bw 65
    max_bw 65
  ]
  edge [
    source 26
    target 88
    bw 92
    max_bw 92
  ]
  edge [
    source 26
    target 89
    bw 73
    max_bw 73
  ]
  edge [
    source 27
    target 41
    bw 77
    max_bw 77
  ]
  edge [
    source 27
    target 52
    bw 69
    max_bw 69
  ]
  edge [
    source 27
    target 56
    bw 65
    max_bw 65
  ]
  edge [
    source 27
    target 63
    bw 95
    max_bw 95
  ]
  edge [
    source 27
    target 68
    bw 77
    max_bw 77
  ]
  edge [
    source 27
    target 69
    bw 54
    max_bw 54
  ]
  edge [
    source 27
    target 79
    bw 74
    max_bw 74
  ]
  edge [
    source 27
    target 94
    bw 71
    max_bw 71
  ]
  edge [
    source 27
    target 98
    bw 62
    max_bw 62
  ]
  edge [
    source 28
    target 31
    bw 69
    max_bw 69
  ]
  edge [
    source 28
    target 35
    bw 55
    max_bw 55
  ]
  edge [
    source 28
    target 40
    bw 92
    max_bw 92
  ]
  edge [
    source 28
    target 46
    bw 62
    max_bw 62
  ]
  edge [
    source 28
    target 50
    bw 90
    max_bw 90
  ]
  edge [
    source 28
    target 54
    bw 52
    max_bw 52
  ]
  edge [
    source 28
    target 59
    bw 91
    max_bw 91
  ]
  edge [
    source 28
    target 60
    bw 78
    max_bw 78
  ]
  edge [
    source 28
    target 64
    bw 84
    max_bw 84
  ]
  edge [
    source 28
    target 71
    bw 71
    max_bw 71
  ]
  edge [
    source 28
    target 75
    bw 95
    max_bw 95
  ]
  edge [
    source 28
    target 78
    bw 73
    max_bw 73
  ]
  edge [
    source 28
    target 79
    bw 77
    max_bw 77
  ]
  edge [
    source 28
    target 80
    bw 79
    max_bw 79
  ]
  edge [
    source 28
    target 96
    bw 58
    max_bw 58
  ]
  edge [
    source 29
    target 33
    bw 82
    max_bw 82
  ]
  edge [
    source 29
    target 47
    bw 78
    max_bw 78
  ]
  edge [
    source 29
    target 57
    bw 78
    max_bw 78
  ]
  edge [
    source 29
    target 60
    bw 68
    max_bw 68
  ]
  edge [
    source 29
    target 68
    bw 93
    max_bw 93
  ]
  edge [
    source 29
    target 70
    bw 98
    max_bw 98
  ]
  edge [
    source 29
    target 74
    bw 54
    max_bw 54
  ]
  edge [
    source 29
    target 86
    bw 88
    max_bw 88
  ]
  edge [
    source 29
    target 96
    bw 82
    max_bw 82
  ]
  edge [
    source 29
    target 97
    bw 90
    max_bw 90
  ]
  edge [
    source 30
    target 33
    bw 82
    max_bw 82
  ]
  edge [
    source 30
    target 48
    bw 87
    max_bw 87
  ]
  edge [
    source 30
    target 59
    bw 59
    max_bw 59
  ]
  edge [
    source 30
    target 60
    bw 57
    max_bw 57
  ]
  edge [
    source 30
    target 67
    bw 70
    max_bw 70
  ]
  edge [
    source 30
    target 68
    bw 74
    max_bw 74
  ]
  edge [
    source 30
    target 70
    bw 86
    max_bw 86
  ]
  edge [
    source 30
    target 73
    bw 99
    max_bw 99
  ]
  edge [
    source 30
    target 86
    bw 99
    max_bw 99
  ]
  edge [
    source 30
    target 90
    bw 55
    max_bw 55
  ]
  edge [
    source 30
    target 94
    bw 63
    max_bw 63
  ]
  edge [
    source 30
    target 99
    bw 55
    max_bw 55
  ]
  edge [
    source 31
    target 44
    bw 74
    max_bw 74
  ]
  edge [
    source 31
    target 46
    bw 80
    max_bw 80
  ]
  edge [
    source 31
    target 52
    bw 56
    max_bw 56
  ]
  edge [
    source 31
    target 57
    bw 73
    max_bw 73
  ]
  edge [
    source 31
    target 77
    bw 51
    max_bw 51
  ]
  edge [
    source 31
    target 80
    bw 78
    max_bw 78
  ]
  edge [
    source 31
    target 82
    bw 81
    max_bw 81
  ]
  edge [
    source 31
    target 90
    bw 96
    max_bw 96
  ]
  edge [
    source 31
    target 92
    bw 80
    max_bw 80
  ]
  edge [
    source 32
    target 42
    bw 64
    max_bw 64
  ]
  edge [
    source 32
    target 44
    bw 95
    max_bw 95
  ]
  edge [
    source 32
    target 46
    bw 95
    max_bw 95
  ]
  edge [
    source 32
    target 52
    bw 84
    max_bw 84
  ]
  edge [
    source 32
    target 53
    bw 91
    max_bw 91
  ]
  edge [
    source 32
    target 69
    bw 50
    max_bw 50
  ]
  edge [
    source 32
    target 76
    bw 70
    max_bw 70
  ]
  edge [
    source 33
    target 48
    bw 82
    max_bw 82
  ]
  edge [
    source 33
    target 52
    bw 64
    max_bw 64
  ]
  edge [
    source 33
    target 55
    bw 68
    max_bw 68
  ]
  edge [
    source 33
    target 67
    bw 88
    max_bw 88
  ]
  edge [
    source 33
    target 69
    bw 96
    max_bw 96
  ]
  edge [
    source 33
    target 86
    bw 84
    max_bw 84
  ]
  edge [
    source 33
    target 92
    bw 86
    max_bw 86
  ]
  edge [
    source 34
    target 59
    bw 66
    max_bw 66
  ]
  edge [
    source 34
    target 68
    bw 73
    max_bw 73
  ]
  edge [
    source 34
    target 96
    bw 63
    max_bw 63
  ]
  edge [
    source 35
    target 46
    bw 72
    max_bw 72
  ]
  edge [
    source 35
    target 52
    bw 99
    max_bw 99
  ]
  edge [
    source 35
    target 64
    bw 65
    max_bw 65
  ]
  edge [
    source 35
    target 65
    bw 64
    max_bw 64
  ]
  edge [
    source 35
    target 74
    bw 91
    max_bw 91
  ]
  edge [
    source 36
    target 46
    bw 69
    max_bw 69
  ]
  edge [
    source 36
    target 55
    bw 70
    max_bw 70
  ]
  edge [
    source 36
    target 68
    bw 52
    max_bw 52
  ]
  edge [
    source 36
    target 77
    bw 92
    max_bw 92
  ]
  edge [
    source 36
    target 79
    bw 93
    max_bw 93
  ]
  edge [
    source 36
    target 95
    bw 52
    max_bw 52
  ]
  edge [
    source 36
    target 96
    bw 58
    max_bw 58
  ]
  edge [
    source 37
    target 49
    bw 52
    max_bw 52
  ]
  edge [
    source 38
    target 60
    bw 76
    max_bw 76
  ]
  edge [
    source 38
    target 83
    bw 62
    max_bw 62
  ]
  edge [
    source 39
    target 86
    bw 79
    max_bw 79
  ]
  edge [
    source 40
    target 42
    bw 59
    max_bw 59
  ]
  edge [
    source 40
    target 54
    bw 74
    max_bw 74
  ]
  edge [
    source 40
    target 57
    bw 82
    max_bw 82
  ]
  edge [
    source 40
    target 74
    bw 89
    max_bw 89
  ]
  edge [
    source 40
    target 76
    bw 80
    max_bw 80
  ]
  edge [
    source 40
    target 89
    bw 52
    max_bw 52
  ]
  edge [
    source 40
    target 96
    bw 99
    max_bw 99
  ]
  edge [
    source 41
    target 69
    bw 51
    max_bw 51
  ]
  edge [
    source 41
    target 80
    bw 73
    max_bw 73
  ]
  edge [
    source 42
    target 44
    bw 87
    max_bw 87
  ]
  edge [
    source 42
    target 51
    bw 57
    max_bw 57
  ]
  edge [
    source 42
    target 58
    bw 54
    max_bw 54
  ]
  edge [
    source 42
    target 79
    bw 99
    max_bw 99
  ]
  edge [
    source 42
    target 91
    bw 63
    max_bw 63
  ]
  edge [
    source 42
    target 98
    bw 94
    max_bw 94
  ]
  edge [
    source 43
    target 56
    bw 90
    max_bw 90
  ]
  edge [
    source 43
    target 84
    bw 81
    max_bw 81
  ]
  edge [
    source 44
    target 53
    bw 84
    max_bw 84
  ]
  edge [
    source 44
    target 61
    bw 73
    max_bw 73
  ]
  edge [
    source 44
    target 63
    bw 73
    max_bw 73
  ]
  edge [
    source 44
    target 98
    bw 56
    max_bw 56
  ]
  edge [
    source 45
    target 59
    bw 67
    max_bw 67
  ]
  edge [
    source 45
    target 61
    bw 64
    max_bw 64
  ]
  edge [
    source 45
    target 85
    bw 99
    max_bw 99
  ]
  edge [
    source 45
    target 93
    bw 88
    max_bw 88
  ]
  edge [
    source 45
    target 94
    bw 100
    max_bw 100
  ]
  edge [
    source 45
    target 95
    bw 75
    max_bw 75
  ]
  edge [
    source 46
    target 55
    bw 80
    max_bw 80
  ]
  edge [
    source 46
    target 59
    bw 99
    max_bw 99
  ]
  edge [
    source 46
    target 68
    bw 80
    max_bw 80
  ]
  edge [
    source 46
    target 72
    bw 86
    max_bw 86
  ]
  edge [
    source 46
    target 79
    bw 92
    max_bw 92
  ]
  edge [
    source 46
    target 96
    bw 99
    max_bw 99
  ]
  edge [
    source 46
    target 99
    bw 91
    max_bw 91
  ]
  edge [
    source 47
    target 54
    bw 89
    max_bw 89
  ]
  edge [
    source 47
    target 71
    bw 51
    max_bw 51
  ]
  edge [
    source 47
    target 77
    bw 58
    max_bw 58
  ]
  edge [
    source 47
    target 84
    bw 57
    max_bw 57
  ]
  edge [
    source 47
    target 92
    bw 100
    max_bw 100
  ]
  edge [
    source 47
    target 99
    bw 78
    max_bw 78
  ]
  edge [
    source 48
    target 60
    bw 53
    max_bw 53
  ]
  edge [
    source 48
    target 66
    bw 71
    max_bw 71
  ]
  edge [
    source 48
    target 71
    bw 86
    max_bw 86
  ]
  edge [
    source 48
    target 76
    bw 79
    max_bw 79
  ]
  edge [
    source 48
    target 78
    bw 51
    max_bw 51
  ]
  edge [
    source 48
    target 92
    bw 99
    max_bw 99
  ]
  edge [
    source 48
    target 95
    bw 81
    max_bw 81
  ]
  edge [
    source 49
    target 54
    bw 81
    max_bw 81
  ]
  edge [
    source 49
    target 59
    bw 68
    max_bw 68
  ]
  edge [
    source 49
    target 62
    bw 89
    max_bw 89
  ]
  edge [
    source 49
    target 80
    bw 66
    max_bw 66
  ]
  edge [
    source 49
    target 81
    bw 61
    max_bw 61
  ]
  edge [
    source 49
    target 92
    bw 63
    max_bw 63
  ]
  edge [
    source 49
    target 99
    bw 95
    max_bw 95
  ]
  edge [
    source 50
    target 52
    bw 65
    max_bw 65
  ]
  edge [
    source 50
    target 64
    bw 70
    max_bw 70
  ]
  edge [
    source 50
    target 67
    bw 66
    max_bw 66
  ]
  edge [
    source 50
    target 72
    bw 80
    max_bw 80
  ]
  edge [
    source 50
    target 77
    bw 96
    max_bw 96
  ]
  edge [
    source 51
    target 56
    bw 83
    max_bw 83
  ]
  edge [
    source 51
    target 59
    bw 55
    max_bw 55
  ]
  edge [
    source 51
    target 65
    bw 54
    max_bw 54
  ]
  edge [
    source 51
    target 69
    bw 76
    max_bw 76
  ]
  edge [
    source 51
    target 73
    bw 70
    max_bw 70
  ]
  edge [
    source 51
    target 89
    bw 94
    max_bw 94
  ]
  edge [
    source 51
    target 98
    bw 70
    max_bw 70
  ]
  edge [
    source 52
    target 65
    bw 95
    max_bw 95
  ]
  edge [
    source 52
    target 80
    bw 76
    max_bw 76
  ]
  edge [
    source 52
    target 89
    bw 52
    max_bw 52
  ]
  edge [
    source 53
    target 96
    bw 90
    max_bw 90
  ]
  edge [
    source 54
    target 59
    bw 68
    max_bw 68
  ]
  edge [
    source 54
    target 65
    bw 72
    max_bw 72
  ]
  edge [
    source 54
    target 67
    bw 67
    max_bw 67
  ]
  edge [
    source 54
    target 69
    bw 99
    max_bw 99
  ]
  edge [
    source 54
    target 88
    bw 68
    max_bw 68
  ]
  edge [
    source 54
    target 92
    bw 54
    max_bw 54
  ]
  edge [
    source 55
    target 77
    bw 64
    max_bw 64
  ]
  edge [
    source 55
    target 79
    bw 61
    max_bw 61
  ]
  edge [
    source 55
    target 80
    bw 76
    max_bw 76
  ]
  edge [
    source 55
    target 84
    bw 96
    max_bw 96
  ]
  edge [
    source 55
    target 96
    bw 60
    max_bw 60
  ]
  edge [
    source 56
    target 60
    bw 51
    max_bw 51
  ]
  edge [
    source 56
    target 70
    bw 71
    max_bw 71
  ]
  edge [
    source 56
    target 80
    bw 71
    max_bw 71
  ]
  edge [
    source 56
    target 86
    bw 50
    max_bw 50
  ]
  edge [
    source 56
    target 92
    bw 55
    max_bw 55
  ]
  edge [
    source 57
    target 60
    bw 97
    max_bw 97
  ]
  edge [
    source 57
    target 77
    bw 83
    max_bw 83
  ]
  edge [
    source 57
    target 78
    bw 54
    max_bw 54
  ]
  edge [
    source 57
    target 80
    bw 82
    max_bw 82
  ]
  edge [
    source 57
    target 92
    bw 95
    max_bw 95
  ]
  edge [
    source 57
    target 95
    bw 94
    max_bw 94
  ]
  edge [
    source 57
    target 97
    bw 81
    max_bw 81
  ]
  edge [
    source 57
    target 99
    bw 56
    max_bw 56
  ]
  edge [
    source 58
    target 63
    bw 80
    max_bw 80
  ]
  edge [
    source 58
    target 76
    bw 99
    max_bw 99
  ]
  edge [
    source 58
    target 82
    bw 62
    max_bw 62
  ]
  edge [
    source 58
    target 91
    bw 50
    max_bw 50
  ]
  edge [
    source 58
    target 98
    bw 72
    max_bw 72
  ]
  edge [
    source 59
    target 63
    bw 51
    max_bw 51
  ]
  edge [
    source 59
    target 68
    bw 50
    max_bw 50
  ]
  edge [
    source 59
    target 69
    bw 50
    max_bw 50
  ]
  edge [
    source 59
    target 70
    bw 55
    max_bw 55
  ]
  edge [
    source 59
    target 80
    bw 71
    max_bw 71
  ]
  edge [
    source 59
    target 94
    bw 69
    max_bw 69
  ]
  edge [
    source 59
    target 98
    bw 58
    max_bw 58
  ]
  edge [
    source 60
    target 69
    bw 70
    max_bw 70
  ]
  edge [
    source 60
    target 70
    bw 85
    max_bw 85
  ]
  edge [
    source 60
    target 95
    bw 84
    max_bw 84
  ]
  edge [
    source 60
    target 96
    bw 84
    max_bw 84
  ]
  edge [
    source 61
    target 80
    bw 85
    max_bw 85
  ]
  edge [
    source 61
    target 93
    bw 74
    max_bw 74
  ]
  edge [
    source 61
    target 94
    bw 94
    max_bw 94
  ]
  edge [
    source 62
    target 73
    bw 63
    max_bw 63
  ]
  edge [
    source 62
    target 74
    bw 77
    max_bw 77
  ]
  edge [
    source 62
    target 87
    bw 68
    max_bw 68
  ]
  edge [
    source 62
    target 91
    bw 91
    max_bw 91
  ]
  edge [
    source 63
    target 72
    bw 93
    max_bw 93
  ]
  edge [
    source 63
    target 75
    bw 78
    max_bw 78
  ]
  edge [
    source 63
    target 82
    bw 67
    max_bw 67
  ]
  edge [
    source 63
    target 95
    bw 75
    max_bw 75
  ]
  edge [
    source 64
    target 67
    bw 99
    max_bw 99
  ]
  edge [
    source 64
    target 73
    bw 63
    max_bw 63
  ]
  edge [
    source 64
    target 79
    bw 93
    max_bw 93
  ]
  edge [
    source 64
    target 93
    bw 95
    max_bw 95
  ]
  edge [
    source 65
    target 68
    bw 75
    max_bw 75
  ]
  edge [
    source 65
    target 81
    bw 74
    max_bw 74
  ]
  edge [
    source 65
    target 86
    bw 80
    max_bw 80
  ]
  edge [
    source 65
    target 97
    bw 64
    max_bw 64
  ]
  edge [
    source 66
    target 67
    bw 82
    max_bw 82
  ]
  edge [
    source 66
    target 68
    bw 52
    max_bw 52
  ]
  edge [
    source 66
    target 70
    bw 90
    max_bw 90
  ]
  edge [
    source 66
    target 77
    bw 62
    max_bw 62
  ]
  edge [
    source 66
    target 85
    bw 59
    max_bw 59
  ]
  edge [
    source 66
    target 92
    bw 82
    max_bw 82
  ]
  edge [
    source 66
    target 97
    bw 61
    max_bw 61
  ]
  edge [
    source 66
    target 99
    bw 60
    max_bw 60
  ]
  edge [
    source 67
    target 71
    bw 52
    max_bw 52
  ]
  edge [
    source 67
    target 72
    bw 99
    max_bw 99
  ]
  edge [
    source 67
    target 76
    bw 84
    max_bw 84
  ]
  edge [
    source 68
    target 75
    bw 95
    max_bw 95
  ]
  edge [
    source 68
    target 92
    bw 100
    max_bw 100
  ]
  edge [
    source 69
    target 87
    bw 78
    max_bw 78
  ]
  edge [
    source 69
    target 88
    bw 99
    max_bw 99
  ]
  edge [
    source 70
    target 77
    bw 51
    max_bw 51
  ]
  edge [
    source 70
    target 98
    bw 95
    max_bw 95
  ]
  edge [
    source 71
    target 92
    bw 90
    max_bw 90
  ]
  edge [
    source 72
    target 92
    bw 86
    max_bw 86
  ]
  edge [
    source 74
    target 87
    bw 84
    max_bw 84
  ]
  edge [
    source 74
    target 99
    bw 72
    max_bw 72
  ]
  edge [
    source 75
    target 80
    bw 82
    max_bw 82
  ]
  edge [
    source 75
    target 86
    bw 83
    max_bw 83
  ]
  edge [
    source 75
    target 90
    bw 65
    max_bw 65
  ]
  edge [
    source 76
    target 80
    bw 72
    max_bw 72
  ]
  edge [
    source 76
    target 88
    bw 66
    max_bw 66
  ]
  edge [
    source 77
    target 78
    bw 76
    max_bw 76
  ]
  edge [
    source 77
    target 92
    bw 76
    max_bw 76
  ]
  edge [
    source 78
    target 99
    bw 84
    max_bw 84
  ]
  edge [
    source 79
    target 82
    bw 55
    max_bw 55
  ]
  edge [
    source 79
    target 86
    bw 65
    max_bw 65
  ]
  edge [
    source 79
    target 92
    bw 84
    max_bw 84
  ]
  edge [
    source 79
    target 93
    bw 67
    max_bw 67
  ]
  edge [
    source 79
    target 95
    bw 98
    max_bw 98
  ]
  edge [
    source 80
    target 85
    bw 64
    max_bw 64
  ]
  edge [
    source 80
    target 98
    bw 83
    max_bw 83
  ]
  edge [
    source 82
    target 93
    bw 56
    max_bw 56
  ]
  edge [
    source 82
    target 95
    bw 62
    max_bw 62
  ]
  edge [
    source 83
    target 93
    bw 57
    max_bw 57
  ]
  edge [
    source 84
    target 99
    bw 78
    max_bw 78
  ]
  edge [
    source 85
    target 86
    bw 86
    max_bw 86
  ]
  edge [
    source 85
    target 92
    bw 91
    max_bw 91
  ]
  edge [
    source 85
    target 99
    bw 68
    max_bw 68
  ]
  edge [
    source 86
    target 96
    bw 79
    max_bw 79
  ]
  edge [
    source 89
    target 95
    bw 50
    max_bw 50
  ]
  edge [
    source 89
    target 96
    bw 93
    max_bw 93
  ]
  edge [
    source 91
    target 93
    bw 91
    max_bw 91
  ]
  edge [
    source 92
    target 98
    bw 91
    max_bw 91
  ]
  edge [
    source 93
    target 99
    bw 98
    max_bw 98
  ]
  edge [
    source 94
    target 97
    bw 68
    max_bw 68
  ]
  edge [
    source 98
    target 99
    bw 89
    max_bw 89
  ]
]
