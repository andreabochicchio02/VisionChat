document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('mic-btn');
    const chatLog = document.getElementById('chat-log');

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('msg', sender);
        
        messageDiv.textContent = text;

        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
        return messageDiv;
    }

    async function handleListenClick() {
        btn.disabled = true;
        btn.classList.add('listening');
        btn.innerHTML = "";

        try {
            const response = await fetch('/listen', { method: 'POST' });

            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            if (!response.body) {
                throw new Error('Streaming not supported (response.body is null)');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            // Keep this outside the loop so we continue appending across chunks
            let currentStreamMessage = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);

                        if (data.user) {
                            addMessage(data.user, 'user');
                        }

                        if (data.assistant) {
                            // a full assistant message arrived; show it and reset stream state
                            addMessage(data.assistant, 'ai');
                            currentStreamMessage = null;
                        }

                        if (data.token) {
                            // ensure we have a single streaming AI message to append to
                            if (!currentStreamMessage) {
                                currentStreamMessage = addMessage("", "ai");
                            }
                            // SAFELY append token as text node (no innerHTML)
                            currentStreamMessage.appendChild(document.createTextNode(data.token));
                            chatLog.scrollTop = chatLog.scrollHeight;
                        }

                    } catch (e) {
                        console.error('Error parsing JSON:', e, line);
                    }
                }
            }

            // Reset button state
            btn.disabled = false;
            btn.classList.remove('listening');
            btn.innerHTML = "🎤";

        } catch (error) {
            console.error('Error connecting to backend:', error);
            addMessage("Error connecting to the server.", "ai");

            btn.disabled = false;
            btn.classList.remove('listening');
            btn.innerHTML = "🎤";
        }
    }

    // attach handler
    btn.addEventListener('click', handleListenClick);

    // Notifications
    let eventSource = null;

    function setupNotifications() {
        if (eventSource) {
            eventSource.close();
        }

        eventSource = new EventSource('/notifications');

        eventSource.onopen = () => {
            console.log('SSE connected');
        };

        eventSource.onmessage = (event) => {

            if (!event.data || event.data.trim() === '') return;

            try {
                const data = JSON.parse(event.data);
                if (data.notification) {
                    addMessage(data.notification, 'ai');
                }
            } catch (e) {
                console.error('Error parsing notification:', e);
            }
        };

        eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
        };

        console.log('Notification listener started');
    }

    setupNotifications();

    window.addEventListener('beforeunload', () => {
        if (eventSource) {
            eventSource.close();
        }
    });

});
