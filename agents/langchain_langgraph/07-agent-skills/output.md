 python3 agent.py

════════════════════════════════════════════════════════════
USER: Write a SQL query to find the top 5 customers by total revenue in the last 90 days, with order count and average order value. Tables: orders(id, customer_id, total, created_at), customers(id, name, email).
────────────────────────────────────────────────────────────
  🔧 load_skill(skill_name='write_sql')
  🔧 load_skill(skill_name='write_sql')

Here is the SQL query to find the top 5 customers by total revenue in the last 90 days, along with their order count and average order value:

```sql
-- Brief explanation of the approach:
-- This query uses a Common Table Expression (CTE) to filter orders from the last 90 days and calculate total revenue, order count, and average order value for each customer.
-- It then selects the top 5 customers by total revenue and includes their order count and average order value.

WITH recent_orders AS (
    SELECT 
        o.customer_id,
        SUM(o.total) AS total_revenue,
        COUNT(o.id) AS order_count,
        AVG(o.total) AS average_order_value
    FROM 
        orders o
    WHERE 
        o.created_at >= NOW() - INTERVAL '90 days'
    GROUP BY 
        o.customer_id
),
ranked_customers AS (
    SELECT 
        c.name,
        c.email,
        ro.total_revenue,
        ro.order_count,
        ro.average_order_value,
        RANK() OVER (ORDER BY ro.total_revenue DESC) AS revenue_rank
    FROM 
        recent_orders ro
    JOIN 
        customers c ON ro.customer_id = c.id
)
SELECT 
    name,
    email,
    total_revenue,
    order_count,
    average_order_value
FROM 
    ranked_customers
WHERE 
    revenue_rank <= 5
ORDER BY 
    total_revenue DESC;

-- Performance notes:
-- This query uses window functions and CTEs for better performance and readability.
-- Ensure indexes on orders.created_at and orders.customer_id for efficient filtering and joining.

-- Alternative approaches:
-- A simpler approach without CTEs could be used, but it might be less readable and performant.
```

This query will provide the top 5 customers by total revenue in the last 90 days, along with their order count and average order value.


════════════════════════════════════════════════════════════
USER: Review this clause: 'The Vendor shall not be liable for any indirect, incidental, or consequential damages. Total liability shall not exceed fees paid in the last 30 days. Agreement auto-renews annually unless cancelled with 90 days written notice.'
────────────────────────────────────────────────────────────
  🔧 load_skill(skill_name='review_legal_doc')
  🔧 load_skill(skill_name='review_legal_doc')

<thinking> The clause has been reviewed, and the results have been compiled. I will now provide the final review to the user. </thinking>

**Clause Review:**

1. **Document Summary:**
   - Type: Contract Clause
   - Parties: Vendor and Customer
   - Purpose: To define liability, auto-renewal, and cancellation terms

2. **🔴 High Risk Clauses:**
   - Limitation of liability: "The Vendor shall not be liable for any indirect, incidental, or consequential damages. Total liability shall not exceed fees paid in the last 30 days."
     - This clause significantly limits the Vendor's liability, which may be unfavorable to the Customer. It is recommended to review this clause with a lawyer to ensure it aligns with the Customer's risk tolerance and contractual expectations.

3. **🟡 Medium Risk Clauses:**
   - Auto-renewal and cancellation notice: "Agreement auto-renews annually unless cancelled with 90 days written notice."
     - The 90-day written notice period for cancellation may be considered long. It is worth negotiating this term to reduce the notice period to a more reasonable timeframe, such as 30 or 60 days.

4. **🟢 Standard / Acceptable Clauses:**
   - No additional standard clauses were identified in the provided text.

5. **Missing Clauses:**
   - Intellectual property assignment and licensing
   - Payment terms and late payment penalties
   - Data privacy and confidentiality obligations
   - Force majeure and dispute resolution

6. **Overall Risk Rating: MEDIUM**

**Recommendation:**
It is advisable for the Customer to consult with a lawyer to review the high-risk clauses and negotiate the medium-risk clauses. Additionally, the Customer should ensure that missing standard protections are included in the contract to safeguard their interests.


════════════════════════════════════════════════════════════
USER: Review this Python function:

def get_user(user_id):
    conn = psycopg2.connect('postgresql://admin:password123@db:5432/prod')
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cur.fetchone()
────────────────────────────────────────────────────────────
  🔧 load_skill(skill_name='code_review')
  🔧 load_skill(skill_name='code_review')

<thinking> The tool result confirms the rules and output format for the code review. I will now proceed with the code review based on these guidelines. </thinking>

**Language:** Python

**Summary:** The function `get_user` retrieves a user from a PostgreSQL database based on the provided `user_id`. The function establishes a connection to the database, executes a SQL query, and returns the result. 

**🔴 Critical Issues:**
1. **SQL Injection Vulnerability:** The function uses string formatting to construct the SQL query, which is vulnerable to SQL injection. This is a severe security risk.
2. **Hardcoded Secrets:** The database connection string contains hardcoded credentials (`admin:password123`), which is insecure.
3. **Lack of Error Handling:** The function does not handle potential errors that may occur during the database connection or query execution.
4. **Resource Leak:** The function does not close the database connection or cursor, which can lead to resource leaks.

**🟡 Improvements:**
1. **Use Parameterized Queries:** To prevent SQL injection, use parameterized queries instead of string formatting.
2. **Environment Variables for Secrets:** Store database credentials in environment variables or a secure secrets manager.
3. **Error Handling:** Add try-except blocks to handle potential errors and ensure the connection is closed properly.
4. **Context Manager:** Use a context manager to handle the database connection and ensure it is closed automatically.

**🟢 Good Practices:**
1. **Function Purpose:** The function clearly aims to retrieve a user based on their ID, which is a common and useful operation.

**Suggested Refactor:**

```python
import psycopg2
from psycopg2 import sql
import os

def get_user(user_id):
    try:
        # Use environment variables for secrets
        conn = psycopg2.connect('postgresql://{}:{}@db:5432/prod'.format(
            os.getenv('DB_USER'), os.getenv('DB_PASSWORD')
        ))
        with conn.cursor() as cur:
            # Use parameterized query to prevent SQL injection
            query = sql.SQL("SELECT * FROM users WHERE id = %s")
            cur.execute(query, (user_id,))
            return cur.fetchone()
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        if conn:
            conn.close()
```

This refactor addresses the critical issues by using parameterized queries, storing secrets securely, adding error handling, and ensuring the connection is closed properly.


════════════════════════════════════════════════════════════
USER: What can you help me with?
────────────────────────────────────────────────────────────
  🔧 list_skills()

I can help you with the following tasks:

1. **Code Review**: I can review source code in any programming language to identify issues, suggest improvements, and ensure best practices are followed.

2. **Legal Document Review**: I can review contracts, clauses, or any legal text to identify potential issues, ensure compliance, and provide insights on legal risks.

3. **SQL Query Writing**: I can write or generate SQL queries to help you retrieve, manipulate, and analyze data from databases.

Feel free to ask for help with any of these tasks!