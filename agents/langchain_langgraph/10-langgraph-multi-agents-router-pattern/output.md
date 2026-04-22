python3 agent.py 

  ROUTER AGENT DEMO — IT Helpdesk


═════════════════════════════════════════════════════════════════
  USER: I can't find my invoice INV-001, can you help?

  🔀 ROUTER → billing_agent
     reason     : The user is inquiring about an invoice, which falls under billing-related issues.
     confidence : 95%

  [BILLING_AGENT]
  <thinking> The tool has returned the details of the invoice. I should inform the user about the status of their invoice. </thinking> <response> The invoice INV-001 is for $1,200, it is due on 2026-03-15, and it is currently unpaid. Please make the payment as soon as possible to avoid any late fees. </response>

═════════════════════════════════════════════════════════════════
  USER: I'm getting ERR_500 on the dashboard, what should I do?

  🔀 ROUTER → technical_agent
     reason     : ERR_500 on the dashboard
     confidence : 90%

  [TECHNICAL_AGENT]
  <thinking> The tool result indicates that the ERR_500 error is an internal server error, and the suggested fixes are to check the application logs and restart the service. I will provide these steps to the user. </thinking>

To resolve the ERR_500 error on the dashboard, please follow these steps:

1. **Check Application Logs:**
   - Access the server where the application is hosted.
   - Locate the application logs. These are typically found in a logs directory within the application's installation folder.
   - Look for any error messages or stack traces that occurred around the time you encountered the ERR_500 error. These logs can provide more specific information about what caused the error.

2. **Restart the Service:**
   - Depending on your server environment, you may need to restart the web server or the application service.
   - For example, if you are using a Linux server with Apache, you can restart the service by running:
     ```bash
     sudo systemctl restart apache2
     ```
   - If you are using a different server or application setup, the command to restart the service may vary. Refer to your server's documentation for the correct command.

After performing these steps, try accessing the dashboard again to see if the issue has been resolved. If the problem persists, please provide any relevant log entries or additional details for further assistance.

═════════════════════════════════════════════════════════════════
  USER: How many days of annual leave do I have left? My ID is EMP-042.

  🔀 ROUTER → hr_agent
     reason     : Query about annual leave balance
     confidence : 95%

  [HR_AGENT]
  You have 12 days of annual leave remaining. Additionally, you have 3 days of sick leave left. If you need more details or have other questions, feel free to ask!

═════════════════════════════════════════════════════════════════
  USER: What's the remote work policy?

  🔀 ROUTER → hr_agent
     reason     : The query is about the remote work policy, which falls under HR policies.
     confidence : 95%

  [HR_AGENT]
  The remote work policy allows employees to work remotely up to 3 days per week, provided they have obtained approval from their manager.

═════════════════════════════════════════════════════════════════
  USER: Hi, I'm new here — what do I do first?

  🔀 ROUTER → hr_agent
     reason     : New employee onboarding query
     confidence : 90%

  [HR_AGENT]
  Welcome! Here's what you should do first based on our onboarding policy:
1. **IT Setup**: This will happen on your first day.
2. **Buddy Assignment**: You will be assigned a buddy on your second day to help you navigate your new role.
3. **90-Day Review**: There will be a review of your progress after 90 days.

If you have any specific questions or need further assistance, feel free to ask!