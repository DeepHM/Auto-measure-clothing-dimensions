

def getSizingPts(cloth_cat_id):
    # ================================== 
    # annos['annotations']
    # ================================== 
    # * index for clothes category 
    # 1: short sleeve top 
    # 2: long sleeve top 
    # 7: shorts 
    # 8: Trousers 
    # 9: Skirt 
    # ... etc. 
    # ================================== 
    
    if cloth_cat_id == 1: # 1: short sleeve top 
        # *** T-Shirts ***
        # ---------------------------- 
        measure_index = {
            'a': 'Total-length',   # 총장 
            'b': 'Chest-size',     # 가슴둘레 
            'c': 'Waist-length',   # 허리둘레
            'd': 'Arm-width',      # 팔둘레
        }

        measure_points = {
            'a': [1, 16],   # 총장 
            # 'b': [12, 20],  # 가슴둘레 # armfit
            'b': [13, 19],  # 가슴둘레 # beside chest 
            'c': [15, 17],  # 허리둘레
            'd': [9, 10],   # 팔둘레
        }

        # *** ver 1. 
        # measure_points = {
        #     'a': [1, 20],   # 총장 
        #     'b': [17, 23],  # 가슴둘레 
        #     'c': [7, 33],   # 어깨너비
        #     'd': [18, 22],  # 허리둘레
        # }

    elif cloth_cat_id == 2: # 2: long sleeve top 
        # *** Long Sleeve top ***
        # ---------------------------- 
        measure_index = {
            'a': 'Total-length',   # 총장 
            'b': 'Chest-size',     # 가슴둘레 
            'c': 'Waist-length',   # 허리둘레
            'd': 'Wrist-hole',     # 손목둘레
            'e': 'Arm-length',     # 소매길이 
            'f': 'Neck-hole',      # 목 둘레
        }

        measure_points = {
            'a': [1, 20],     # 총장 
            # 'b': [16, 24],    # (upper)가슴둘레 
            'b': [17, 23],    # (below)가슴둘레 
            'c': [19, 21],    # 허리둘레
            'd': [11, 12],    # 손목둘레
            'e': [11, 7],     # 소매길이 
            'f': [2, 6],      # 목 둘레   
        }
    elif cloth_cat_id == 7: # 7: shorts 
        # *** Shorts ***
        # (cf1. 총장(바지길이: pants length))
        # (cf2. 허벅지 둘레: thigh width는 밑위 바로 옆으로 측정 길이 임))
        # ---------------------------- 
        measure_index = {
            'a': 'Waist-length',     # 허리둘레
            'b': 'Pants-length',     # 총장 (=바지길이: pants length) 
            'c': 'Hole-width',       # 밑단 
            # 'd': 'Waist-to-crotch',  # 밑위 (: 미정)
            # 'e': 'Thigh-width',      # 허벅지 둘레 (: 미정)
        }

        measure_points = {
            'a': [1, 3],    # 허리둘레
            'b': [1, 5],    # 총장 (=바지길이: pants length) 
            'c': [5, 6],    # 밑단 
            # 'd': [2, 7],    # 밑위
            # 'e': [4, 7],    # 허벅지 둘레 
        }

    elif cloth_cat_id == 8: # 8: Trousers         
        # *** pants ***
        # ---------------------------- 
        measure_index = {
            'a': 'Waist-length',     # 허리둘레
            'b': 'Leg-length',       # 다리소매길이 
            'c': 'Ankle-width',      # 발목너비 
            'd': 'Crotch-length',    # 밑위길이 
        }

        measure_points = {
            'a': [1, 3],    # 허리둘레
            'b': [1, 6],    # 다리소매길이 
            'c': [6, 7],    # 발목너비 
            'd': [2, 9],    # 밑위길이 
        }
        
    elif cloth_cat_id == 9: # 9: Skirt 
        # *** skirts ***
        # ---------------------------- 
        measure_index = {
            'a': 'Waist-length',     # 허리둘레
            'b': 'Total-length',     # 총장 
        }

        measure_points = {
            'a': [1, 3],    # 허리둘레
            'b': [2, 6],    # 총장 
        }

    return(measure_index, measure_points)