map = {
    "semantickitti" : {
        "labels_name" :['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'],
        "source_to_common" :{
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
            15:15,
            16:16,
            17:17,
            18:18
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
            15:15,
            16:16,
            17:17,
            18:18
        }
    },
    "nuscenes": {
        "labels_name" : ["Motorcycle", "Bicycle", "Pedestrian", "Driveable Ground", "Sidewalk", "Other Ground", "Manmade", "Vegetation", "4-Wheeled", "Terrain"],
        "source_to_common":{
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
        "target_to_common":{
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
    },
    "pandaset": {
        "labels_name":["2-wheeled", "Pedestrian", "Driveable Ground", "Sidewalk", "Other Ground", "Manmade", "Vegetation", "4-Wheeled"],
        "source_to_common":{
                -1:-1,
                0:7,
                1:0,
                2:0,
                4:7,
                3:7,
                5:1,
                6:0,
                7:0,
                8:2,
                9:2,
                10:3,
                11:4,
                12:5,
                13:5,
                14:6,
                15:6,
                16:4,
                17:6,
                18:6
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
    },
    "semanticposs":{
        "labels_name":[ "person", "rider", "2-wheeled", "4-wheeled", "ground", "trunk", "vegetation", "traffic-sign", "pole", "building", "fence"],
        "source_to_common":{
            -1:-1,
            0:3,
            1:2,
            2:2,
            4:3,
            3:3,
            5:0,
            6:1,
            7:1,
            8:4,
            9:4,
            10:4,
            11:4,
            12:9,
            13:10,
            14:6,
            15:5,
            16:4,
            17:8,
            18:7
        },
        "target_to_common":{
            -1:-1,
            0:0,
            1:1,
            11:2,
            2:3,
            12:4,
            3: 5,
            4:6,
            5:7,
            6:8,
            8:9,
            10:10,
            7:9,
            9:9
    
        }
    },

    "parisluco3d":{
        "labels_name" :['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'],
        "source_to_common" :{
            -1:-1,
            0:0,
            1:1,
            2:2,
            3:3,
            4:4,
            5:5,
            6:1,
            7:2,
            8:6,
            9:7,
            10:8,
            11:9,
            12:10,
            13:11,
            14:12,
            15:13,
            16:14,
            17:15,
            18:16
        },
            "target_to_common":{
                -1:-1,
                0:0,
                1:1,
                2:4,
                3:2,
                4:2,
                5:3,
                6:4,
                7:4,
                8:5,
                9:6,
                10:6,
                11:6, 
                12:7, 
                13:6, 
                14:6, 
                15:9, 
                16:8, 
                17:9, 
                18:10, 
                19:11, 
                20:15, 
                21:16, 
                22:-1, 
                23:16, 
                24:15, 
                25:-1, 
                26:11, 
                27:11, 
                28:12, 
                29:13, 
                30:14, 
                31:11, 
                32:-1, 
                33:-1, 
                34:-1, 
                35:-1, 
                36:-1, 
                37:-1, 
                38:-1, 
                39:-1, 
                40:-1, 
                41:-1
            },
        },
}