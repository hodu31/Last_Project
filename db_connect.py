import mysql.connector

def insert_or_update_data(V_ID, action_id):
    config = {
        'user': 'root',
        'password': '0000',
        'host': '127.0.0.1',
        'database': 'security',
        'port': 3306,
        'raise_on_warnings': True
    }

    conn = None
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()

        sql = """
        INSERT INTO LASTPRO_VISIT (V_ID, action_id)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE
            SHOP_ID = VALUES(SHOP_ID),
            V_ID = VALUES(V_ID),
            V_CAM = VALUES(V_CAM);
        """
        cursor.execute(sql, (V_ID, action_id))
        conn.commit()

    except mysql.connector.Error as err:
        print(f"오류: {err}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# 함수 사용 예시
insert_or_update_data('user1', 'shop1', 'visit1', 'binary_data1')











import mysql.connector

def update_database(pred):
    config = {
        'user': 'root',
        'password': '0000',
        'host': '127.0.0.1',
        'database': 'security',
        'port': 3306,
        'raise_on_warnings': True
    }

    conn = None
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()

        for user_id, actions in pred.items():
            for action_code, status in actions.items():
                if status:  # Only consider if status is True
                    # Check the current status in the database
                    cursor.execute("SELECT status FROM ACTIONS WHERE id=%s AND action_code=%s", (user_id, action_code))
                    result = cursor.fetchone()

                    # If the record doesn't exist or its status is False, then update
                    if not result or not result[0]:
                        sql = """
                        INSERT INTO ACTIONS (id, action_code, status)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            status = VALUES(status);
                        """
                        cursor.execute(sql, (user_id, action_code, status))
                        conn.commit()

    except mysql.connector.Error as err:
        print(f"오류: {err}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Sample pred dictionary
pred = {
    'user1': {
        'action_code1': True,
        'action_code2': False
    },
    'user2': {
        'action_code1': False,
        'action_code2': True
    }
}

# Update the database
update_database(pred)
