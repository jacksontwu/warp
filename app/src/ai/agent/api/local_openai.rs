use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, anyhow};
use futures::channel::oneshot;
use futures_lite::stream;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;
use warp_multi_agent_api as api;

use super::convert_to::convert_input;
use super::{RequestParams, ResponseStream};
use crate::ai_assistant::AIGeneratedCommand;
use crate::ai_assistant::execution_context::WarpAiExecutionContext;
use crate::ai_assistant::utils::TranscriptPart;
use crate::drive::workflows::ai_assist::{GeneratedArgument, GeneratedCommandMetadata};
use crate::server::server_api::AIApiError;

const SKIP_LOGIN_ENV: &str = "WARP_SKIP_LOGIN";
const API_KEY_ENV: &str = "WARP_LOCAL_LLM_API_KEY";
const BASE_URL_ENV: &str = "WARP_LOCAL_LLM_BASE_URL";
const MODEL_ENV: &str = "WARP_LOCAL_LLM_MODEL";

#[derive(Clone, Debug, PartialEq, Eq)]
struct LocalOpenAIConfig {
    api_key: String,
    chat_completions_url: String,
    model: String,
}

impl LocalOpenAIConfig {
    fn from_env() -> anyhow::Result<Self> {
        let api_key = env_value(API_KEY_ENV, "OPENAI_API_KEY")?;
        let base_url = env_value(BASE_URL_ENV, "OPENAI_BASE_URL")?;
        let model = env_value(MODEL_ENV, "OPENAI_MODEL")?;
        Self::from_values(api_key, base_url, model)
    }

    fn from_values(api_key: String, base_url: String, model: String) -> anyhow::Result<Self> {
        let mut url = url::Url::parse(base_url.trim()).context("invalid local LLM base URL")?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(anyhow!("local LLM base URL must use HTTP or HTTPS"));
        }
        if url.host_str().is_none() {
            return Err(anyhow!("local LLM base URL must include a host"));
        }
        if !url.path().ends_with("/chat/completions") {
            let path = format!("{}/chat/completions", url.path().trim_end_matches('/'));
            url.set_path(&path);
        }
        if api_key.trim().is_empty() {
            return Err(anyhow!("local LLM API key must not be empty"));
        }
        if model.trim().is_empty() {
            return Err(anyhow!("local LLM model must not be empty"));
        }
        Ok(Self {
            api_key,
            chat_completions_url: url.into(),
            model,
        })
    }
}

pub(crate) fn prompt_blocking(prompt: &str) -> anyhow::Result<String> {
    if !mode_enabled() {
        return Err(anyhow!("set {SKIP_LOGIN_ENV}=1 to use the local LLM"));
    }
    if prompt.trim().is_empty() {
        return Err(anyhow!("local LLM prompt must not be empty"));
    }
    let config = LocalOpenAIConfig::from_env()?;
    let response = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .context("failed to initialize local LLM client")?
        .post(&config.chat_completions_url)
        .bearer_auth(&config.api_key)
        .json(&ChatRequest {
            model: config.model,
            messages: vec![json!({ "role": "user", "content": prompt })],
            stream: false,
        })
        .send()
        .context("local LLM request failed")?;
    let status = response.status();
    let body = response
        .text()
        .context("failed to read local LLM response")?;
    if !status.is_success() {
        return Err(anyhow!(
            "local LLM returned HTTP {status}: {}",
            error_message(&body)
        ));
    }
    response_text(&body)
}

pub(super) fn mode_enabled() -> bool {
    std::env::var(SKIP_LOGIN_ENV).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn env_value(primary: &str, fallback: &str) -> anyhow::Result<String> {
    std::env::var(primary)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::env::var(fallback)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .ok_or_else(|| anyhow!("set {primary} (or {fallback}) when {SKIP_LOGIN_ENV}=1"))
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct GeneratedCommandsResponse {
    commands: Vec<LocalGeneratedCommand>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct LocalGeneratedCommand {
    command: String,
    #[serde(default)]
    description: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct LocalGeneratedCommandMetadata {
    command: String,
    title: String,
    description: String,
    #[serde(default)]
    arguments: Vec<LocalGeneratedArgument>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct LocalGeneratedArgument {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    default_value: String,
}

async fn send_chat_request(
    config: &LocalOpenAIConfig,
    messages: Vec<Value>,
) -> anyhow::Result<String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .context("failed to initialize local LLM client")?;
    let response = client
        .post(&config.chat_completions_url)
        .bearer_auth(&config.api_key)
        .json(&ChatRequest {
            model: config.model.clone(),
            messages,
            stream: false,
        })
        .send()
        .await
        .context("local LLM request failed")?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read local LLM response")?;
    if !status.is_success() {
        return Err(anyhow!(
            "local LLM returned HTTP {status}: {}",
            error_message(&body)
        ));
    }
    response_text(&body)
}

pub(super) async fn generate_commands(
    prompt: &str,
    execution_context: Option<&WarpAiExecutionContext>,
) -> anyhow::Result<Vec<AIGeneratedCommand>> {
    if prompt.trim().is_empty() {
        return Err(anyhow!("local LLM command prompt must not be empty"));
    }
    let config = LocalOpenAIConfig::from_env()?;
    let environment = execution_context
        .and_then(WarpAiExecutionContext::to_json_string)
        .unwrap_or_else(|| "unknown".to_owned());
    let output = send_chat_request(
        &config,
        vec![
            json!({
                "role": "system",
                "content": format!(
                    "Translate the user's request into one or more shell commands for this environment: {environment}. Return only strict JSON with this schema: {{\"commands\":[{{\"command\":\"...\",\"description\":\"...\"}}]}}. Do not use Markdown fences."
                ),
            }),
            json!({ "role": "user", "content": prompt }),
        ],
    )
    .await?;

    let commands = parse_generated_commands(&output)?;
    if commands.is_empty() {
        return Err(anyhow!("local LLM returned no generated commands"));
    }
    let commands = commands
        .into_iter()
        .filter_map(|command| {
            let value = command.command.trim();
            if value.is_empty() {
                return None;
            }
            let description = if command.description.trim().is_empty() {
                prompt.trim().to_owned()
            } else {
                command.description.trim().to_owned()
            };
            Some(AIGeneratedCommand::new(value.to_owned(), description))
        })
        .collect::<Vec<_>>();
    if commands.is_empty() {
        return Err(anyhow!("local LLM returned no usable generated commands"));
    }
    Ok(commands)
}

pub(super) async fn generate_dialogue(
    transcript: &[TranscriptPart],
    prompt: &str,
    execution_context: Option<&WarpAiExecutionContext>,
) -> anyhow::Result<String> {
    if prompt.trim().is_empty() {
        return Err(anyhow!("local LLM dialogue prompt must not be empty"));
    }
    let config = LocalOpenAIConfig::from_env()?;
    let environment = execution_context
        .and_then(WarpAiExecutionContext::to_json_string)
        .unwrap_or_else(|| "unknown".to_owned());
    let mut messages = vec![json!({
        "role": "system",
        "content": format!(
            "You are a terminal assistant. Answer for this execution environment: {environment}. Give concise, actionable explanations and shell commands."
        ),
    })];
    for part in transcript {
        messages.push(json!({ "role": "user", "content": part.raw_user_prompt() }));
        messages.push(json!({ "role": "assistant", "content": part.raw_assistant_answer() }));
    }
    messages.push(json!({ "role": "user", "content": prompt }));
    send_chat_request(&config, messages).await
}

pub(super) async fn generate_command_metadata(
    command: &str,
) -> anyhow::Result<GeneratedCommandMetadata> {
    if command.trim().is_empty() {
        return Err(anyhow!("local LLM metadata command must not be empty"));
    }
    let config = LocalOpenAIConfig::from_env()?;
    let output = send_chat_request(
        &config,
        vec![
            json!({
                "role": "system",
                "content": "Describe and parameterize the shell command. Return only strict JSON with this schema: {\"command\":\"...\",\"title\":\"...\",\"description\":\"...\",\"arguments\":[{\"name\":\"...\",\"description\":\"...\",\"default_value\":\"...\"}]}. Do not use Markdown fences.",
            }),
            json!({ "role": "user", "content": command }),
        ],
    )
    .await?;
    let metadata = parse_json_response::<LocalGeneratedCommandMetadata>(&output)?;
    if metadata.command.trim().is_empty() || metadata.title.trim().is_empty() {
        return Err(anyhow!("local LLM returned incomplete command metadata"));
    }
    Ok(GeneratedCommandMetadata {
        command: metadata.command,
        title: metadata.title,
        description: metadata.description,
        arguments: metadata
            .arguments
            .into_iter()
            .map(|argument| GeneratedArgument {
                name: argument.name,
                description: argument.description,
                default_value: argument.default_value,
            })
            .collect(),
    })
}

fn parse_generated_commands(output: &str) -> anyhow::Result<Vec<LocalGeneratedCommand>> {
    Ok(parse_json_response::<GeneratedCommandsResponse>(output)?.commands)
}

fn parse_json_response<T: for<'de> Deserialize<'de>>(output: &str) -> anyhow::Result<T> {
    let trimmed = output.trim();
    let json = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|value| value.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(trimmed);
    serde_json::from_str(json).context("local LLM returned invalid JSON")
}

pub(super) async fn generate(
    params: RequestParams,
    mut cancellation_rx: oneshot::Receiver<()>,
) -> ResponseStream {
    let result = generate_events(params, &mut cancellation_rx).await;
    let events = match result {
        Ok(events) => events.into_iter().map(Ok).collect(),
        Err(error) => vec![Err(Arc::new(AIApiError::Other(error)))],
    };
    Box::pin(stream::iter(events))
}

async fn generate_events(
    mut params: RequestParams,
    cancellation_rx: &mut oneshot::Receiver<()>,
) -> anyhow::Result<Vec<api::ResponseEvent>> {
    let config = LocalOpenAIConfig::from_env()?;
    let api_input =
        convert_input(std::mem::take(&mut params.input)).context("unsupported local LLM input")?;
    let latest_queries = queries_from_input(&api_input);
    if latest_queries.is_empty() {
        return Err(anyhow!(
            "local LLM mode currently supports user queries, not Warp server-managed tool continuations"
        ));
    }

    let mut messages = vec![json!({
        "role": "system",
        "content": "You are a coding assistant running locally in Warp. Answer directly and accurately. Warp server tools are unavailable in local mode, so do not claim to have run commands or changed files."
    })];
    append_task_history(&mut messages, &params.tasks);
    messages.extend(
        latest_queries
            .iter()
            .map(|query| json!({ "role": "user", "content": query })),
    );

    let response = tokio::select! {
        response = send_chat_request(&config, messages) => response?,
        _ = cancellation_rx => return Err(anyhow!("local LLM request cancelled")),
    };

    Ok(response_events(
        params,
        latest_queries,
        response,
        config.model,
    ))
}

fn response_text(body: &str) -> anyhow::Result<String> {
    let response: ChatResponse =
        serde_json::from_str(body).context("invalid local LLM Chat Completions response")?;
    let message = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("local LLM response contained no choices"))?
        .message;
    message
        .content
        .filter(|content| !content.is_empty())
        .or(message.reasoning_content)
        .filter(|content| !content.is_empty())
        .ok_or_else(|| anyhow!("local LLM response contained no text"))
}

fn append_task_history(messages: &mut Vec<Value>, tasks: &[api::Task]) {
    for task in tasks {
        for message in &task.messages {
            match message.message.as_ref() {
                Some(api::message::Message::UserQuery(query)) => {
                    messages.push(json!({ "role": "user", "content": query.query }));
                }
                Some(api::message::Message::AgentOutput(output)) => {
                    messages.push(json!({ "role": "assistant", "content": output.text }));
                }
                _ => {}
            }
        }
    }
}

fn queries_from_input(input: &api::request::Input) -> Vec<String> {
    use api::request::input::Type;
    use api::request::input::user_inputs::user_input::Input;

    match input.r#type.as_ref() {
        Some(Type::UserInputs(inputs)) => inputs
            .inputs
            .iter()
            .filter_map(|input| match input.input.as_ref() {
                Some(Input::UserQuery(query)) => Some(query.query.clone()),
                Some(Input::CliAgentUserQuery(query)) => {
                    query.user_query.as_ref().map(|query| query.query.clone())
                }
                _ => None,
            })
            .collect(),
        #[allow(deprecated)]
        Some(Type::UserQuery(query)) => vec![query.query.clone()],
        Some(Type::QueryWithCannedResponse(query)) => vec![query.query.clone()],
        Some(Type::AutoCodeDiffQuery(query)) => vec![query.query.clone()],
        Some(Type::CreateNewProject(query)) => vec![query.query.clone()],
        Some(Type::SummarizeConversation(query)) => vec![query.prompt.clone()],
        _ => Vec::new(),
    }
}

fn response_events(
    params: RequestParams,
    latest_queries: Vec<String>,
    output: String,
    model: String,
) -> Vec<api::ResponseEvent> {
    use api::client_action::Action;
    use api::response_event::Type;

    let request_id = Uuid::new_v4().to_string();
    let conversation_id = params
        .conversation_token
        .as_ref()
        .map(|token| token.as_str().to_owned())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let task_id = params
        .tasks
        .iter()
        .find(|task| task.dependencies.is_none())
        .or_else(|| params.tasks.first())
        .map(|task| task.id.clone())
        .unwrap_or_else(|| format!("local-root-{conversation_id}"));

    let mut actions = Vec::new();
    if params.tasks.is_empty() {
        actions.push(api::ClientAction {
            action: Some(Action::CreateTask(api::client_action::CreateTask {
                task: Some(api::Task {
                    id: task_id.clone(),
                    ..Default::default()
                }),
            })),
        });
    }
    let mut task_messages = latest_queries
        .into_iter()
        .map(|query| api::Message {
            id: Uuid::new_v4().to_string(),
            task_id: task_id.clone(),
            request_id: request_id.clone(),
            message: Some(api::message::Message::UserQuery(api::message::UserQuery {
                query,
                ..Default::default()
            })),
            ..Default::default()
        })
        .collect::<Vec<_>>();
    task_messages.push(api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.clone(),
        request_id: request_id.clone(),
        message: Some(api::message::Message::ModelUsed(api::message::ModelUsed {
            model_id: model.clone(),
            model_display_name: model,
            is_fallback: false,
            prompt_cache_expires_at: None,
        })),
        ..Default::default()
    });
    task_messages.push(api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.clone(),
        request_id: request_id.clone(),
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput { text: output },
        )),
        ..Default::default()
    });
    actions.push(api::ClientAction {
        action: Some(Action::AddMessagesToTask(
            api::client_action::AddMessagesToTask {
                task_id,
                messages: task_messages,
            },
        )),
    });

    vec![
        api::ResponseEvent {
            r#type: Some(Type::Init(api::response_event::StreamInit {
                conversation_id,
                request_id,
                run_id: String::new(),
            })),
        },
        api::ResponseEvent {
            r#type: Some(Type::ClientActions(api::response_event::ClientActions {
                actions,
            })),
        },
        api::ResponseEvent {
            r#type: Some(Type::Finished(api::response_event::StreamFinished {
                reason: Some(api::response_event::stream_finished::Reason::Done(
                    api::response_event::stream_finished::Done {},
                )),
                ..Default::default()
            })),
        },
    ]
}

fn error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| body.chars().take(500).collect())
}

#[cfg(test)]
#[path = "local_openai_tests.rs"]
mod tests;
