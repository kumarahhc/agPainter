import os
import io
import sys
import base64
from typing import Annotated

# The OpenAI SDK's response parser uses recursive type construction that can exceed
# Python's default limit (1000) on large responses with many tool calls.
sys.setrecursionlimit(5000)

import PIL.Image
import PIL.ImageDraw

from autogen import AssistantAgent, UserProxyAgent
import dotenv
dotenv.load_dotenv()

#------------------------------
# Configurations
#------------------------------
API_PROXY   = os.getenv("API_PROXY", "proxy-not-set")
MODEL       = "openai/gpt-4.1-mini"
TOPIC       = "A beach scene with coconut trees, a sun, and a boat on the water."
output_dir  = "output"
NUM_ROUNDS  = 10
CANVAS_SIZE = (200, 200)

_config     = [{"model": MODEL, "base_url": API_PROXY, "api_key": "not-required"}]
LLM_Painter = {"config_list": _config, "cache_seed": None, "temperature": 0.35, "max_tokens": 2048}
LLM_Critic  = {"config_list": _config, "cache_seed": None, "temperature": 0.6,  "max_tokens": 2048}

#------------------------------
# Canvas
#------------------------------
class Canvas:
    def __init__(self, size):
        self.image = PIL.Image.new("RGB", size, (255, 255, 255))
        self.draw  = PIL.ImageDraw.Draw(self.image)
        self.round = 0

    def save(self):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"round_{self.round}.png")
        self.image.save(path)
        print(f"  [Canvas] Saved → {path}")

    def to_base64(self) -> str:
        buf = io.BytesIO()
        self.image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


canvas = Canvas(CANVAS_SIZE)

#------------------------------
# Drawing Tools
# Annotated provides information to AG2 to auto-generate the JSON schema sent to the LLM.
# Return values (text) confirms what was done in wach call.
#------------------------------
def draw_line(
    x1:    Annotated[int, "Start x coordinate (0-199)"],
    y1:    Annotated[int, "Start y coordinate (0-199)"],
    x2:    Annotated[int, "End x coordinate (0-199)"],
    y2:    Annotated[int, "End y coordinate (0-199)"],
    color: Annotated[str, "Color name or hex, e.g. 'blue' or '#0000FF'"] = "black",
    width: Annotated[int, "Line width in pixels"] = 1,
) -> str:
    canvas.draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    return f"Drew line ({x1},{y1})->({x2},{y2}) color={color} width={width}"

def draw_circle(
    x:      Annotated[int, "Center x coordinate"],
    y:      Annotated[int, "Center y coordinate"],
    radius: Annotated[int, "Radius in pixels"],
    color:  Annotated[str, "Outline color"] = "black",
    fill:   Annotated[str, "Fill color, or 'none' for hollow"] = "none",
    width:  Annotated[int, "Outline width in pixels"] = 1,
) -> str:
    fc = None if fill == "none" else fill
    canvas.draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)],
                        outline=color, fill=fc, width=width)
    return f"Drew circle at ({x},{y}) r={radius} color={color} fill={fill}"

def draw_rectangle(
    x1:    Annotated[int, "Top-left x"],
    y1:    Annotated[int, "Top-left y"],
    x2:    Annotated[int, "Bottom-right x"],
    y2:    Annotated[int, "Bottom-right y"],
    color: Annotated[str, "Outline color"] = "black",
    fill:  Annotated[str, "Fill color, or 'none' for hollow"] = "none",
    width: Annotated[int, "Outline width in pixels"] = 1,
) -> str:
    fc = None if fill == "none" else fill
    canvas.draw.rectangle([(x1, y1), (x2, y2)], outline=color, fill=fc, width=width)
    return f"Drew rect ({x1},{y1})->({x2},{y2}) color={color} fill={fill}"

def draw_point(
    x:      Annotated[int, "x coordinate"],
    y:      Annotated[int, "y coordinate"],
    color:  Annotated[str, "Pixel color"] = "black",
) -> str:
    canvas.draw.point([(x, y)], fill=color)
    return f"Drew pixel at ({x},{y}) color={color}"

def draw_polygon(
    points: Annotated[str, "Flat comma-separated x,y pairs: 'x1,y1,x2,y2,x3,y3,...' (at least 3 points)"],
    color:  Annotated[str, "Outline color"] = "black",
    fill:   Annotated[str, "Fill color, or 'none' for hollow"] = "none",
    width:  Annotated[int, "Outline width in pixels"] = 1,
) -> str:
    coords = list(map(int, points.split(",")))
    pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    fc = None if fill == "none" else fill
    canvas.draw.polygon(pts, outline=color, fill=fc, width=width)
    return f"Drew polygon {pts} color={color} fill={fill}"
#------------------------------
# Painter Agent
#------------------------------
painter = AssistantAgent(
    name="Painter",
    llm_config=LLM_Painter,
    system_message=(
        f"You are a digital painter. Draw on a {CANVAS_SIZE[0]}x{CANVAS_SIZE[1]} canvas.\n"
        f"Topic: {TOPIC}\n"
        "Use draw_line, draw_circle, draw_rectangle, draw_point to paint the scene. "
        "Make many calls to add detail. "
        "When you have finished all drawing calls for this round, reply with exactly: DRAWING_COMPLETE"
    ),
)
# Same fix as executor: remove the built-in auto-reply counter so the Painter
painter._reply_func_list = [
    f for f in painter._reply_func_list
    if f.get("reply_func").__name__ not in (
        "check_termination_and_human_reply",
        "a_check_termination_and_human_reply",
    )
]

#------------------------------
# Critic Agent
#------------------------------
critic = AssistantAgent(
    name="Critic",
    llm_config=LLM_Critic,
    system_message=(
        f"You are an art critic evaluating a digital painting.\n"
        f"Topic: {TOPIC}\n"        
        "You will receive the actual rendered image. Base your feedback on what you see visually.\n"
        "Provide structured feedback:\n"
        "1. What works well\n"
        "2. What is missing\n"
        "3. Specific improvements to be made in the next round"
    ),
)

#------------------------------
# Executor
# Runs drawing tool calls proposed by the Painter.
# Also manages round transitions
#------------------------------
executor = UserProxyAgent(
    name="Executor",
    human_input_mode="NEVER",
    is_termination_msg=lambda _: False,
    code_execution_config=False,
)
# Remove the built-in auto-reply counter check entirely.
executor._reply_func_list = [
    f for f in executor._reply_func_list
    if f.get("reply_func").__name__ not in (
        "check_termination_and_human_reply",
        "a_check_termination_and_human_reply",
    )
]

#------------------------------
# Critic Proxy
# Sends the canvas image to the Critic and collects structured feedback.
#------------------------------
critic_proxy = UserProxyAgent(
    name="CriticProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,   # one-shot: send image, receive feedback
    code_execution_config=False,
)

# Register tools: Painter proposes calls and Executor runs them
for fn in [draw_line, draw_circle, draw_rectangle, draw_point]:
    painter.register_for_llm()(fn)
    executor.register_for_execution()(fn)

#------------------------------
# Reply counter (position=0 — fires first)
#------------------------------
_reply_count = 0

def _count_replies(*args, **kwargs):
    global _reply_count
    _reply_count += 1
    messages  = kwargs.get("messages", args[1] if len(args) > 1 else [])
    last      = messages[-1] if messages else {}
    tool_name = last.get("name", "")
    label     = f" → {tool_name}" if tool_name else ""
    print(f"  [Reply #{_reply_count}{label}]")
    return False, None  # observe only; do not override

executor.register_reply(trigger=painter, reply_func=_count_replies, position=0)


_completed_rounds = 0

def handle_round_transition(*args, **kwargs):
    global _completed_rounds, _reply_count

    messages     = kwargs.get("messages", args[1] if len(args) > 1 else [])
    last_content = (messages[-1].get("content") or "") if messages else ""

    if "DRAWING_COMPLETE" not in last_content:
        return False, None  # normal tool call — let default tool execution handle it

    # Round complete 
    # — save canvas, get Critic feedback, and prepare next round message for Painter
    _completed_rounds += 1
    canvas.round = _completed_rounds
    canvas.save()
    print(f"  Total drawing calls this round: {_reply_count}")
    _reply_count = 0

    # Clear accumulated history so the next round starts with a fresh context.
    executor.chat_messages.clear()
    painter.chat_messages.clear()

    if _completed_rounds >= NUM_ROUNDS:
        print(f"\nAll {NUM_ROUNDS} rounds complete. Final image: "
              f"{output_dir}/round_{canvas.round}.png")
        return True, None   # terminate the chat

    # Get Critic feedback on the rendered image
    img_b64 = canvas.to_base64()
    feedback = ""
    try:
        critic_proxy.initiate_chat(
            critic,
            clear_history=True,
            message={
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": f"Round {_completed_rounds}/{NUM_ROUNDS} painting — please evaluate:"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            },
        )
        last     = critic_proxy.last_message(critic)
        feedback = last.get("content", "") if last else ""
    except Exception as e:
        print(f"  [Critic] ERROR on round {_completed_rounds}: {e}")
        feedback = f"(Critic unavailable this round: {e})"
    feedback_path = os.path.join(output_dir, "feedback.txt")
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(f"=== Round {_completed_rounds} ===\n{feedback}\n\n")
    print(f"  [Critic] Feedback appended → {feedback_path}")
    # Return image + feedback to the Painter for the next round
    return True, {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"Round {_completed_rounds + 1}/{NUM_ROUNDS}.\n"
                    f"Critic feedback on your drawing:\n{feedback}\n\n"
                    "Following is the your current canvas — improve it based on the feedback provided above:"
                ),
            },
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ],
    }

#------------------------------
# Round transition reply (position=1, runs after _count_ replies)
#------------------------------
executor.register_reply(trigger=painter, reply_func=handle_round_transition, position=1)

#------------------------------
# Start the process with initiate_chat
#------------------------------
print(f"Starting {NUM_ROUNDS}-round Painter-Critic session.")
print(f"Topic: {TOPIC}\n")

executor.initiate_chat(
    painter,
    message=(
        f"Round 1/{NUM_ROUNDS}: Paint the following scene on the canvas.\n"
        f"Topic: {TOPIC}\n"
        f"Canvas: {CANVAS_SIZE[0]}x{CANVAS_SIZE[1]} pixels (origin top-left, x to right, y to down).\n"
        "When you have finished all drawing calls for this round, "
        "reply with exactly: DRAWING_COMPLETE"
    ),
    summary_method=None,  # skip end-of-chat summarization; this is used as history is cleared between rounds
)
