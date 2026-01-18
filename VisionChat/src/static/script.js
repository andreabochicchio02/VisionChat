document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('mic-btn');
    const chatLog = document.getElementById('chat-log');

    /**
     * text - The message content
     * sender - 'user' or 'ai' to determine styling
     */
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('msg');
        messageDiv.classList.add(sender); // Adds .user or .ai class
        
        if(sender === 'ai') {
             messageDiv.innerHTML = text; 
        } else {
             messageDiv.textContent = text;
        }

        chatLog.appendChild(messageDiv);
        
        // Auto-scroll to the bottom of the chat container
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    // Setup Server-Sent Events for real-time notifications
    function setupNotifications() {
        const eventSource = new EventSource('/notifications');
        
        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                // Display notification as AI message
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

    // Start listening for notifications
    setupNotifications();

    // Event Listener for the microphone button
    btn.addEventListener('click', async () => {
        // Start listening animation
        btn.disabled = true;
        btn.classList.add('listening');
        btn.innerHTML = ""; 
        
        try {
            // Send POST request to backend with streaming response
            const response = await fetch('/listen', { method: 'POST' });
            
            // Read the response as a stream of text
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            while (true) {
                const { done, value } = await reader.read();
                
                // Continuously reads chunks until the server closes the stream.
                if (done) break;
                
                // Decode the chunk and add to buffer
                buffer += decoder.decode(value, { stream: true });
                
                // Process complete JSON objects (separated by newlines)
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer
                
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const data = JSON.parse(line);
                            
                            // When user text is recognized, display it immediately
                            if (data.user) {
                                // Stop listening animation
                                btn.classList.remove('listening');
                                btn.innerHTML = "🎤";
                                
                                // Add user message to chat
                                addMessage(data.user, 'user');
                            }
                            
                            // When assistant response arrives, display it
                            if (data.assistant) {
                                addMessage(data.assistant, 'ai');
                                
                                // Re-enable microphone after everything is complete
                                btn.disabled = false;
                            }
                            
                        } catch (e) {
                            console.error('Error parsing JSON:', e, line);
                        }
                    }
                }
            }

        } catch (error) {
            // Error handling
            console.error('Error connecting to backend:', error);
            addMessage("Error connecting to the server.", "ai");
            
            // Reset button state on error
            btn.disabled = false;
            btn.classList.remove('listening');
            btn.innerHTML = "🎤";
        }
    });
});