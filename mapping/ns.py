map = {
    "nuscenes":{
        "labels_name": ["barrier","bicycle","bus","car","construction_vehicle","motorcycle","pedestrian","traffic_cone","trailer","truck","driveable_surface","other_flat","sidewalk","terrain","manmade","vegetation"],
        "source_to_common":{
            -1:-1,
            0:0,
            1:1,
            2:2,
            3:3,
            4:4,
            5:5,
            6:6,
            7:7,
            8:8,
            9:9,
            10:10,
            11:11,
            12:12,
            13:13,
            14:14,
            15:15
        },
        "target_to_common":{
            -1:-1,
            0:0,
            1:1,
            2:2,
            3:3,
            4:4,
            5:5,
            6:6,
            7:7,
            8:8,
            9:9,
            10:10,
            11:11,
            12:12,
            13:13,
            14:14,
            15:15
        }
    },
    "semantickitti":{
       "labels_name" : ["Motorcycle", "Bicycle", "Pedestrian", "Driveable Ground", "Sidewalk", "Other Ground", "Manmade", "Vegetation", "4-Wheeled", "Terrain"],
        "source_to_common":{
                -1:-1,
                0:6,
                1:1,
                2:8,
                3:8,
                4:8,
                5:0,
                6:2,
                7:6,
                8:8,
                9:8,
                10:3,
                11:5,
                12:4,
                13:9,
                14:6,
                15:7,

        },
        "target_to_common":{
                0:8,
                1:1,
                2:0,
                4:8,
                3:8,
                5:2,
                6:1,
                7:0,
                8:3,
                9:3,
                10:4,
                11:5,
                12:6,
                13:6,
                14:7,
                15:7,
                16:9,
                17:6,
                18:6
        },
    },
    "semanticposs":{
        "labels_name":[ "person", "2-wheeled", "4-wheeled", "ground", "vegetation", "manmade"],
        "source_to_common":{
            -1:-1,
            0:5,
            1:1,
            2:2,
            3:2,
            4:2,
            5:1,
            6:0,
            7:5,
            8:2,
            9:2,
            10:3,
            11:3,
            12:3,
            13:3,
            14:5,
            15:4,

        },
        "target_to_common":{
            -1:-1,
            0:0,
            1:1,
            11:1,
            2:2,
            12:3,
            3: 4,
            4:4,
            5:5,
            6:5,
            8:5,
            10:5,
            7:5,
            9:5

        },
    },
    "pandaset":{
        "labels_name":["2-wheeled", "Pedestrian", "Driveable Ground", "Sidewalk", "Other Ground", "Manmade", "Vegetation", "4-Wheeled"],
        "source_to_common":{
                -1:-1,
                0:5,
                1:0,
                2:7,
                3:7,
                4:7,
                5:0,
                6:1,
                7:5,
                8:7,
                9:7,
                10:2,
                11:4,
                12:3,
                13:4,
                14:5,
                15:6,
        },
        "target_to_common":{
            -1:-1,
            0:-1,
            1: -1,
            2: -1,
            3: -1,
            4: -1,
            5: 6,
            6: 4,
            7: 2,
            8: 2,
            9: 2,
            10: 2,
            11: 3, 
            12: 2, 
            13: 7, 
            14: 7, 
            15: 7, 
            16: 7, 
            17: 7, 
            18: 0, 
            19: 7, 
            20: 7, 
            21: 0, 
            22: 7, 
            23: 7, 
            24: 0, 
            25: 0, 
            26: 0, 
            27: -1, 
            28: -1, 
            29: -1, 
            30: 1, 
            31: 1, 
            32: -1, 
            33: -1, 
            34: 5, 
            35: 5, 
            36: 5, 
            37: 5, 
            38: 5, 
            39: 5, 
            40: 5,
            41: 5, 
            42: 5

        },
    }

}