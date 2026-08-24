output "ecr_repository_url" {
  value = aws_ecr_repository.agent.repository_url
}

output "memory_id" {
  value = aws_bedrockagentcore_memory.agent.id
}

output "runtime_execution_role_arn" {
  value = aws_iam_role.runtime_execution.arn
}

output "agent_runtime_arn" {
  value = try(aws_bedrockagentcore_agent_runtime.agent[0].agent_runtime_arn, null)
}
