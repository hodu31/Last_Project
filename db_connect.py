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


