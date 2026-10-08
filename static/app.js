/* ---------------------------------------------------------------------------
   LOTR RAG Chat - front end
   Talks to POST /api/chat (Server-Sent Events) and renders the routing trail,
   retrieved chunks and the streamed answer.
   --------------------------------------------------------------------------- */

(function () {
  "use strict";

  var THEME_KEY = "lotrRag.theme";
  var API = {
    health: "/api/health",
    chat: "/api/chat",
  };

  var SEEDS = [
    "What is the story of Bilbo Baggins?",
    "Who is Tom Bombadil, and what is his role?",
    "What happened at the Council of Elrond?",
    "How does a vector database power retrieval?",
  ];

  var el = {
    thread: document.getElementById("thread"),
    form: document.getElementById("composer"),
    input: document.getElementById("input"),
    sendBtn: document.getElementById("sendBtn"),
    stopBtn: document.getElementById("stopBtn"),
    clearBtn: document.getElementById("clearBtn"),
    themeBtn: document.getElementById("themeBtn"),
    serverDot: document.getElementById("serverDot"),
    serverText: document.getElementById("serverText"),
    serverStatus: document.getElementById("serverStatus"),
    modelTag: document.getElementById("modelTag"),
  };

  var inner = document.createElement("div");
  inner.className = "thread-inner";
  el.thread.appendChild(inner);

  var controller = null;
  var streaming = false;

  /* ------------------------------------------------------------------ theme */

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch (err) {
      /* storage unavailable, ignore */
    }
  }

  function initTheme() {
    var stored = null;
    try {
      stored = localStorage.getItem(THEME_KEY);
    } catch (err) {
      stored = null;
    }
    applyTheme(stored === "light" || stored === "dark" ? stored : "dark");
  }

  el.themeBtn.addEventListener("click", function () {
    var next =
      document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
  });

  /* ----------------------------------------------------------------- health */

  function setServer(state, text, title) {
    el.serverDot.setAttribute("data-state", state);
    el.serverText.textContent = text;
    el.serverStatus.setAttribute("title", title || text);
  }

  function checkHealth() {
    fetch(API.health, { cache: "no-store" })
      .then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      })
      .then(function (data) {
        if (data.model) el.modelTag.textContent = data.model;
        if (data.missing_env && data.missing_env.length) {
          setServer(
            "warn",
            "Setup needed: " + data.missing_env.join(", "),
            "Missing environment variables: " + data.missing_env.join(", ")
          );
        } else if (data.store_error) {
          setServer("error", "Knowledge base offline", data.store_error);
        } else if (data.models_error) {
          setServer("error", "Model error", data.models_error);
        } else if (data.ready && data.knowledge_base_ready) {
          setServer("ok", "Knowledge base connected", "PGVector + " + data.model);
        } else if (data.ready) {
          setServer(
            "warn",
            "Connects on first Tolkien question",
            "Model ready; PGVector is built lazily"
          );
        } else {
          setServer("warn", "Connects on first question", "Runtime builds lazily");
        }
      })
      .catch(function (err) {
        setServer("error", "Server unreachable", String(err));
      });
  }

  /* ------------------------------------------------------------ formatting */

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
  }

  function inline(text) {
    return escapeHtml(text)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  }

  function toHtml(text) {
    return text
      .split(/\n{2,}/)
      .map(function (block) {
        return "<p>" + inline(block).replace(/\n/g, "<br>") + "</p>";
      })
      .join("");
  }

  /* ------------------------------------------------------------------- dom */

  function node(tag, className, text) {
    var element = document.createElement(tag);
    if (className) element.className = className;
    if (text != null) element.textContent = text;
    return element;
  }

  function nearBottom() {
    return el.thread.scrollHeight - el.thread.scrollTop - el.thread.clientHeight < 140;
  }

  function scrollDown(force) {
    if (force || nearBottom()) {
      el.thread.scrollTop = el.thread.scrollHeight;
    }
  }

  /* -------------------------------------------------------------- empty state */

  function renderEmpty() {
    inner.innerHTML = "";
    var wrap = node("div", "empty");
    wrap.appendChild(node("div", "eyebrow", "LOTR RAG Chat - built by Claude Daigan"));
    var h2 = document.createElement("h2");
    h2.innerHTML =
      "Ask about Middle-earth, and get answers <em>grounded in the books</em>.";
    wrap.appendChild(h2);
    wrap.appendChild(
      node(
        "p",
        null,
        "Questions about Tolkien's works are routed to a PGVector knowledge base, which retrieves the most relevant passages before the model answers. Everything else goes straight to the model."
      )
    );
    var chips = node("div", "chips");
    SEEDS.forEach(function (seed) {
      var chip = node("button", "chip", seed);
      chip.type = "button";
      chip.addEventListener("click", function () {
        el.input.value = seed;
        autoGrow();
        submit();
      });
      chips.appendChild(chip);
    });
    wrap.appendChild(chips);
    inner.appendChild(wrap);
  }

  /* -------------------------------------------------------------- user bubble */

  function addUser(text) {
    var wrap = node("div", "msg msg-user");
    wrap.appendChild(node("div", "bubble-user", text));
    inner.appendChild(wrap);
    scrollDown(true);
    return wrap;
  }

  /* ---------------------------------------------------------- assistant card */

  function addAssistant() {
    var article = node("article", "msg msg-assistant");

    var head = node("div", "msg-head");
    var left = node("div", "msg-head-left");
    var badge = node("span", "pill");
    badge.hidden = true;
    var who = node("span", "msg-who", "Answer");
    left.appendChild(badge);
    left.appendChild(who);

    var copy = node("button", "copy-btn", "Copy");
    copy.type = "button";
    head.appendChild(left);
    head.appendChild(copy);

    var body = node("div", "msg-body");

    var trail = node("div", "trail");
    var routeStep = makeStep("Routing the question");
    var retrieveStep = makeStep("Retrieving passages");
    var generateStep = makeStep("Writing the answer");
    var retrievalStarted = false;
    var failed = false;
    retrieveStep.el.hidden = true;
    trail.appendChild(routeStep.el);
    trail.appendChild(retrieveStep.el);
    trail.appendChild(generateStep.el);
    body.appendChild(trail);

    var answer = node("div", "answer");
    var thinking = node("div", "thinking");
    thinking.appendChild(node("span", "tick"));
    thinking.appendChild(node("span", null, "Working..."));
    answer.appendChild(thinking);
    body.appendChild(answer);

    var errorBox = null;
    article.appendChild(head);
    article.appendChild(body);
    inner.appendChild(article);
    scrollDown(true);

    var buffer = "";

    return {
      element: article,
      setRoute: function (isLotr, path) {
        badge.hidden = false;
        badge.className = "pill " + (isLotr ? "pill-rag" : "pill-direct");
        badge.textContent = isLotr ? "RAG" : "Direct";
        who.textContent = isLotr ? "Grounded answer" : "Model answer";
        badge.title = path || "";
        setStep(routeStep, "done", isLotr ? "Routed to RAG" : "Routed to model");
        if (isLotr) {
          retrievalStarted = true;
          retrieveStep.el.hidden = false;
          setStep(retrieveStep, "running", "Retrieving passages");
        }
        setStep(generateStep, "running", "Writing the answer");
      },
      setRetrieval: function (count, chunks, actions) {
        setStep(retrieveStep, "done", "Retrieved " + count + " passages");
        appendRetrieval(body, count, chunks, actions);
      },
      push: function (text) {
        if (thinking.parentNode) thinking.parentNode.removeChild(thinking);
        buffer += text;
        answer.innerHTML = toHtml(buffer) + '<span class="cursor"></span>';
        scrollDown(false);
      },
      fail: function (message) {
        failed = true;
        if (thinking.parentNode) thinking.parentNode.removeChild(thinking);
        if (retrievalStarted && retrieveStep.el.getAttribute("data-state") === "running") {
          setStep(retrieveStep, "done", "Retrieval failed");
        }
        setStep(generateStep, "done", "Stopped");
        errorBox = node("div", "error-box", message);
        body.appendChild(errorBox);
        scrollDown(true);
      },
      finish: function (stopped) {
        if (thinking.parentNode) thinking.parentNode.removeChild(thinking);
        answer.innerHTML = buffer ? toHtml(buffer) : "";
        if (!failed) {
          setStep(generateStep, "done", stopped ? "Stopped" : "Answer ready");
        }
        if (!buffer && !errorBox) {
          answer.appendChild(node("p", null, "(no text returned)"));
        }
        copy.dataset.text = buffer;
        scrollDown(true);
      },
    };
  }

  function makeStep(label) {
    var wrap = node("span", "trail-step");
    wrap.setAttribute("data-state", "running");
    wrap.appendChild(node("span", "tick"));
    var text = node("span", null, label);
    wrap.appendChild(text);
    return { el: wrap, text: text };
  }

  function setStep(step, state, label) {
    step.el.setAttribute("data-state", state);
    if (label) step.text.textContent = label;
  }

  function appendRetrieval(body, count, chunks, actions) {
    var details = node("details", "retrieval");
    var summary = node("summary");
    summary.appendChild(node("span", null, "Retrieved passages"));
    summary.appendChild(node("span", "count", count + " chunks from PGVector"));
    details.appendChild(summary);

    var content = node("div", "retrieval-body");
    (chunks || []).forEach(function (chunk, index) {
      var box = node("div", "chunk");
      var meta = chunk.metadata && Object.keys(chunk.metadata).length
        ? Object.keys(chunk.metadata)
            .map(function (key) {
              return key + "=" + chunk.metadata[key];
            })
            .join("  ")
        : "chunk " + (index + 1);
      box.appendChild(node("div", "chunk-meta", meta));
      box.appendChild(node("p", "chunk-text", chunk.content || ""));
      content.appendChild(box);
    });

    if (actions) {
      var foot = node("div", "retrieval-actions");
      var download = node("button", "ghost-btn", "Download chunks (JSON)");
      download.type = "button";
      download.addEventListener("click", function () {
        var blob = new Blob([JSON.stringify(chunks, null, 2)], {
          type: "application/json",
        });
        var url = URL.createObjectURL(blob);
        var link = document.createElement("a");
        link.href = url;
        link.download = "retrieved_chunks.json";
        link.click();
        URL.revokeObjectURL(url);
      });
      foot.appendChild(download);
      content.appendChild(foot);
    }

    details.appendChild(content);
    body.appendChild(details);
  }

  /* ------------------------------------------------------------------ stream */

  function streamChat(message, onFrame, signal) {
    return fetch(API.chat, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message }),
      signal: signal,
    }).then(function (res) {
      if (!res.ok || !res.body) throw new Error("Server responded " + res.status);
      var reader = res.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";

      function pump() {
        return reader.read().then(function (result) {
          if (result.done) return;
          buffer += decoder.decode(result.value, { stream: true });
          var index;
          while ((index = buffer.indexOf("\n\n")) !== -1) {
            var block = buffer.slice(0, index);
            buffer = buffer.slice(index + 2);
            var payload = block
              .split("\n")
              .filter(function (line) {
                return line.indexOf("data:") === 0;
              })
              .map(function (line) {
                return line.slice(5).trim();
              })
              .join("\n");
            if (!payload) continue;
            var frame;
            try {
              frame = JSON.parse(payload);
            } catch (err) {
              continue;
            }
            onFrame(frame);
          }
          return pump();
        });
      }

      return pump();
    });
  }

  /* ------------------------------------------------------------------ submit */

  function setBusy(busy) {
    streaming = busy;
    el.sendBtn.disabled = busy;
    el.stopBtn.hidden = !busy;
  }

  function submit() {
    var message = el.input.value.trim();
    if (!message || streaming) return;

    var empty = inner.querySelector(".empty");
    if (empty) inner.innerHTML = "";

    addUser(message);
    el.input.value = "";
    autoGrow();

    var card = addAssistant();
    setBusy(true);
    controller = new AbortController();

    var pendingChunks = null;

    streamChat(
      message,
      function (frame) {
        if (frame.type === "route") {
          card.setRoute(Boolean(frame.is_lotr), frame.path);
        } else if (frame.type === "retrieval") {
          pendingChunks = frame.chunks;
          card.setRetrieval(frame.count, frame.chunks, true);
        } else if (frame.type === "token") {
          card.push(frame.text || "");
        } else if (frame.type === "error") {
          card.fail(frame.message || "Unknown error");
        } else if (frame.type === "done") {
          card.finish(false);
        }
      },
      controller.signal
    )
      .then(function () {
        card.finish(false);
      })
      .catch(function (err) {
        if (err && err.name === "AbortError") {
          card.finish(true);
        } else {
          card.fail(String((err && err.message) || err));
        }
      })
      .then(function () {
        controller = null;
        setBusy(false);
        checkHealth();
        void pendingChunks;
      });
  }

  el.form.addEventListener("submit", function (event) {
    event.preventDefault();
    submit();
  });

  el.stopBtn.addEventListener("click", function () {
    if (controller) controller.abort();
  });

  el.clearBtn.addEventListener("click", function () {
    if (controller) controller.abort();
    setBusy(false);
    renderEmpty();
  });

  inner.addEventListener("click", function (event) {
    var button = event.target.closest(".copy-btn");
    if (!button) return;
    var text = button.dataset.text || "";
    if (!text) return;
    navigator.clipboard.writeText(text).then(
      function () {
        var original = button.textContent;
        button.textContent = "Copied";
        setTimeout(function () {
          button.textContent = original;
        }, 1400);
      },
      function () {
        button.textContent = "Copy failed";
      }
    );
  });

  /* --------------------------------------------------------------- composer */

  function autoGrow() {
    el.input.style.height = "auto";
    el.input.style.height = Math.min(el.input.scrollHeight, 168) + "px";
  }

  el.input.addEventListener("input", autoGrow);

  el.input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  });

  /* ------------------------------------------------------------------- boot */

  initTheme();
  renderEmpty();
  checkHealth();
  setInterval(checkHealth, 20000);
})();
