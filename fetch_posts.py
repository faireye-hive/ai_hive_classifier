 
import psycopg2
import json

DB_CONFIG = {
    "host": "hafsql-sql.mahdiyari.info",
    "port": 5432,
    "database": "haf_block_log",
    "user": "hafsql_public",
    "password": "hafsql_public"
}

OUTPUT_FILE = "data/raw.json"

def fetch_posts():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    query = """
    SELECT author, permlink, title, body, json_metadata, created
    FROM hafsql."comments"
    WHERE parent_author = ''
    AND created >= NOW() - INTERVAL '1 day'
    """

    cur.execute(query)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows

def main():
    posts = fetch_posts()
    print(f"{len(posts)} posts encontrados.")

    results = []

    for row in posts:
        author, permlink, title, body, metadata, created = row

        results.append({
            "author": author,
            "permlink": permlink,
            "title": title,
            "body": body,
            "json_metadata": metadata,
            "created": str(created)
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)

    print("Posts salvos em data/raw.json")

if __name__ == "__main__":
    main()
