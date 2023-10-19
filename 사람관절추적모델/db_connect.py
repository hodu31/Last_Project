import mysql.connector

def insert_db_data(data):
    # MySQL 연결 정보 설정
    config = {
        'user': 'admin',
        'password': 'noticare',
        'host': 'db-noticare.cvcrdfcptqp8.ap-northeast-2.rds.amazonaws.com',
        'database': 'security',
        'port': 3306,
        'raise_on_warnings': True
    }
    
    # config = {
    #     'user': 'root',
    #     'password': '0000',
    #     'host': '127.0.0.1',
    #     'database': 'security',
    #     'port': 3306,
    #     'raise_on_warnings': True
    # }

    # 데이터베이스에 연결
    cnx = mysql.connector.connect(**config)

    # 커서 객체 생성
    cursor = cnx.cursor()

    # 데이터를 삽입하는 SQL 쿼리 작성
    insert_query = ("INSERT INTO LASTPRO_DAN"
                    "(USER_ID, SHOP_ID, DAN_V_ID, DAN_TIME, DAN_CODE) "
                    "VALUES (%s, %s, %s, %s, %s)")

    # 데이터 삽입
    cursor.execute(insert_query, tuple(data))

    # 변경 사항 커밋
    cnx.commit()

    # 커서 및 연결 닫기
    cursor.close()
    cnx.close()

def insert_visit(data):
    # MySQL 연결 정보 설정
    config = {
        'user': 'admin',
        'password': 'noticare',
        'host': 'db-noticare.cvcrdfcptqp8.ap-northeast-2.rds.amazonaws.com',
        'database': 'security',
        'port': 3306,
        'raise_on_warnings': True
    }
    
    # config = {
    #     'user': 'root',
    #     'password': '0000',
    #     'host': '127.0.0.1',
    #     'database': 'security',
    #     'port': 3306,
    #     'raise_on_warnings': True
    # }
    
    # 데이터베이스에 연결
    cnx = mysql.connector.connect(**config)

    # 커서 객체 생성
    cursor = cnx.cursor()

    # 데이터를 삽입하는 SQL 쿼리 작성
    insert_query = ("INSERT INTO LASTPRO_VISIT "
                    "(USER_ID, SHOP_ID,  V_ID,  V_ENTIME) "
                    "VALUES (%s, %s, %s, %s)")

    # 데이터 삽입
    cursor.execute(insert_query, tuple(data))

    # 변경 사항 커밋
    cnx.commit()

    # 커서 및 연결 닫기
    cursor.close()
    cnx.close()
    
    
    
def insert_vio(data):
    # MySQL 연결 정보 설정
    config = {
        'user': 'admin',
        'password': 'noticare',
        'host': 'db-noticare.cvcrdfcptqp8.ap-northeast-2.rds.amazonaws.com',
        'database': 'security',
        'port': 3306,
        'raise_on_warnings': True
    }
    
    # config = {
    #     'user': 'root',
    #     'password': '0000',
    #     'host': '127.0.0.1',
    #     'database': 'security',
    #     'port': 3306,
    #     'raise_on_warnings': True
    # }
    
    # 데이터베이스에 연결
    cnx = mysql.connector.connect(**config)

    # 커서 객체 생성
    cursor = cnx.cursor()

    # 데이터를 삽입하는 SQL 쿼리 작성
    insert_query = ("INSERT INTO VIOLENCE "
                    "( LEN_PEOPLE,  VIO_TIME, DAN_CODE ) "
                    "VALUES (%s, %s, %s)")

    # 데이터 삽입
    cursor.execute(insert_query, tuple(data))

    # 변경 사항 커밋
    cnx.commit()

    # 커서 및 연결 닫기
    cursor.close()
    cnx.close()