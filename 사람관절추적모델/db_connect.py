import mysql.connector


# MySQL 데이터베이스 연결 설정
config = {
    'user': 'root',
    'password': '0000',
    'host': '127.0.0.1',
    'database': 'security',
    'port': 3306, 
    'raise_on_warnings': True
}

try:
    # MySQL 데이터베이스에 연결
    with mysql.connector.connect(**config) as conn:
        with conn.cursor() as cursor:
            # SQL 쿼리
            sql = "SELECT * FROM lastpro_users;"
            cursor.execute(sql)

            # 데이터베이스에서 데이터 가져오기
            rows = cursor.fetchall()

            # 컬럼 이름 가져오기
            colnames = cursor.description
            cols = [col[0].lower() for col in colnames]
            
            list_dict = []
            for row in rows:
                row_dict = dict(zip(cols, row))
                list_dict.append(row_dict)

            print(list_dict)

except mysql.connector.Error as err:
    print(f"오류: {err}")
