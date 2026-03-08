python agent.py 

═════════════════════════════════════════════════════════════════
USER: Write a SQL query to find the top 5 customers by total revenue in the last 90 days, including their order count and average order value. Tables: orders(id, customer_id, total, created_at), customers(id, name, email).
─────────────────────────────────────────────────────────────────
<thinking> Based on the loaded skill, I will now write the SQL query to find the top 5 customers by total revenue in the last 90 days, including their order count and average order value. </thinking>

1. **Explanation of the Approach:**
   - Use a Common Table Expression (CTE) to filter orders from the last 90 days.
   - Aggregate the total revenue, order count, and average order value for each customer.
   - Join the aggregated results with the customers table to get customer names and emails.
   - Order the results by total revenue in descending order and limit to the top 5 customers.

2. **SQL Query:**
   ```sql
   WITH recent_orders AS (
       SELECT 
           customer_id, 
           SUM(total) AS total_revenue, 
           COUNT(*) AS order_count, 
           AVG(total) AS average_order_value
       FROM 
           orders
       WHERE 
           created_at >= NOW() - INTERVAL '90 days'
       GROUP BY 
           customer_id
   )
   SELECT 
       c.name, 
       c.email, 
       ro.total_revenue, 
       ro.order_count, 
       ro.average_order_value
   FROM 
       recent_orders ro
   JOIN 
       customers c ON ro.customer_id = c.id
   ORDER BY 
       ro.total_revenue DESC
   LIMIT 5;
   ```

3. **Performance Notes:**
   - Ensure there is an index on the `created_at` column in the `orders` table to improve the performance of the date filter.
   - Consider adding an index on `customer_id` in the `orders` table to speed up the grouping and joining operations.

4. **Alternative Approaches:**
   - If the dataset is small, a simpler query without CTEs could be used, but it may be less performant for larger datasets.

═════════════════════════════════════════════════════════════════
USER: Review this contract clause: 'The Vendor shall not be liable for any indirect, incidental, or consequential damages arising from the use of the Service, even if advised of the possibility of such damages. Total liability shall not exceed the fees paid in the last 30 days. This agreement auto-renews annually unless cancelled with 90 days written notice.'
─────────────────────────────────────────────────────────────────
<thinking> Based on the loaded skill prompt, I will now review the provided contract clause according to the specified rules and output format. </thinking>

1. **Document Summary:**
   - **Type:** Contract Clause
   - **Parties:** Vendor and Customer (implied)
   - **Purpose:** To define the Vendor's liability and the terms of the agreement's auto-renewal and cancellation.

2. **🔴 High Risk Clauses (must review with lawyer):**
   - **Liability Limitation:** "The Vendor shall not be liable for any indirect, incidental, or consequential damages arising from the use of the Service, even if advised of the possibility of such damages. Total liability shall not exceed the fees paid in the last 30 days."
     - This clause significantly limits the Vendor's liability, which may not be acceptable to the Customer.
   - **Auto-renewal and Cancellation:** "This agreement auto-renews annually unless cancelled with 90 days written notice."
     - The 90-day written notice requirement for cancellation may be onerous for the Customer.

3. **🟡 Medium Risk Clauses (worth negotiating):**
   - **Liability Cap:** "Total liability shall not exceed the fees paid in the last 30 days."
     - The cap on liability based on recent fees may be negotiable, especially if the fees are low relative to potential damages.

4. **🟢 Standard / Acceptable Clauses:**
   - The clause is standard in limiting liability for indirect damages, which is common in service agreements.

5. **Missing Clauses (protections not present):**
   - **Intellectual Property:** There is no mention of IP ownership or licensing terms.
   - **Termination Rights:** Specific termination rights beyond the cancellation notice are not detailed.
   - **Jurisdiction and Governing Law:** These are not specified in the provided clause.
   - **Data Privacy and Confidentiality:** Obligations regarding data privacy and confidentiality are not mentioned.
   - **Force Majeure:** There is no clause addressing force majeure events.
   - **Dispute Resolution:** No method for dispute resolution is provided.

6. **Overall Risk Rating: MEDIUM**
   - The clause presents some risks, particularly around liability and cancellation terms, but it is not entirely one-sided. However, several standard protections are missing, which increases the overall risk.

**Recommendation:** It is advisable for the Customer to consult with legal counsel to review and potentially negotiate the terms of this clause, especially the liability limitations and cancellation notice period.

═════════════════════════════════════════════════════════════════
USER: Review this Python function:

def get_user(user_id):
    conn = psycopg2.connect('postgresql://admin:password123@db:5432/prod')
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cur.fetchone()
─────────────────────────────────────────────────────────────────
<thinking> I will now conduct a code review of the provided Python function using the loaded code review skill. </thinking>

**Language: Python**

1. **Summary**: 
   The function `get_user` retrieves a user from a PostgreSQL database using a provided `user_id`. The function establishes a connection to the database, executes a SQL query to fetch the user, and returns the result. The overall quality is low due to critical security and performance issues.

2. **🔴 Critical Issues**:
   - **SQL Injection**: The function uses string formatting to insert `user_id` into the SQL query, which is highly vulnerable to SQL injection attacks. This is a severe security flaw.
   - **Hardcoded Secrets**: The database connection string contains hardcoded credentials (`admin:password123`). This is a significant security risk.

3. **🟡 Improvements**:
   - **Parameterized Queries**: Use parameterized queries to prevent SQL injection.
   - **Environment Variables**: Store database credentials in environment variables instead of hardcoding them.
   - **Connection Management**: Use a context manager to ensure the database connection is properly closed.

4. **🟢 Good Practices**:
   - The function is simple and does one thing, which is a good practice.

5. **Suggested Refactor**:
   ```python
   import psycopg2
   import os
   from contextlib import closing

   def get_user(user_id):
       conn_string = os.getenv('DATABASE_URL')
       with closing(psycopg2.connect(conn_string)) as conn:
           with conn.cursor() as cur:
               cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
               return cur.fetchone()
   ```

This refactor addresses the critical issues by using parameterized queries and environment variables for secrets, and improves connection management with context managers.

═════════════════════════════════════════════════════════════════
USER: What can you help me with?
─────────────────────────────────────────────────────────────────
Here are the skills I can help you with:

1. **write_sql**: Expert PostgreSQL query writer with CTEs, window functions, and performance tips.
2. **review_legal_doc**: Commercial contract reviewer that flags risk clauses and missing protections.
3. **code_review**: Principal engineer reviewer for correctness, security, performance, and style.

How can I assist you today?