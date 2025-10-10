#!/usr/bin/env python3
"""
Professional Web-based Configuration UI for Emulation Experiments
Focus: Complete FL framework configuration with preset/custom toggle mode
"""

from flask import Flask, render_template_string, request, jsonify
import yaml
import json
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_DIR = SCRIPT_DIR / "configs"
CONFIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = SCRIPT_DIR / "output"
RESULTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# Read the HTML template from a separate section for clarity
HTML_TEMPLATE = open('template.html', 'r').read() if Path('template.html').exists() else '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>FL Configuration</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f7fa; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        .section { background: white; padding: 24px; margin-bottom: 20px; border-radius: 8px; border: 1px solid #e2e8f0; }
        .section-title { font-size: 18px; font-weight: 600; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 2px solid #e2e8f0; }
        .form-group { margin-bottom: 16px; }
        label { display: block; margin-bottom: 6px; font-weight: 500; font-size: 14px; color: #4a5568; }
        input, select, textarea { width: 100%; padding: 10px; border: 1px solid #cbd5e0; border-radius: 6px; font-size: 14px; }
        input:focus, select:focus, textarea:focus { outline: none; border-color: #3182ce; box-shadow: 0 0 0 3px rgba(49,130,206,0.1); }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
        .grid-4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 16px; }
        .client-card { background: white; border: 2px solid #3182ce; border-radius: 8px; padding: 20px; margin-bottom: 16px; }
        .client-header { display: flex; justify-content: space-between; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 2px solid #e2e8f0; }
        .client-title { font-size: 16px; font-weight: 600; }
        .badge { background: #3182ce; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 500; }
        .badge-server { background: #805ad5; }
        .btn { padding: 10px 20px; border: none; border-radius: 6px; font-size: 14px; font-weight: 500; cursor: pointer; }
        .btn-primary { background: #3182ce; color: white; }
        .btn-primary:hover { background: #2c5282; }
        .btn-secondary { background: #718096; color: white; }
        .btn-secondary:hover { background: #4a5568; }
        .clients-container { min-height: 100px; max-height: 800px; overflow-y: auto; padding: 16px; border: 3px dashed #cbd5e0; border-radius: 8px; background: #f7fafc; }
        .help-text { font-size: 12px; color: #718096; margin-top: 4px; }
        .notification { position: fixed; top: 20px; right: 20px; padding: 14px 20px; background: white; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); transform: translateX(400px); transition: transform 0.3s; z-index: 1000; }
        .notification.active { transform: translateX(0); }
        .notification.success { border-left: 4px solid #48bb78; }
        .notification.error { border-left: 4px solid #f56565; }
        .switch-label { display: flex; align-items: center; margin: 0; font-weight: 500; font-size: 14px; cursor: pointer; }
        .switch { position: relative; display: inline-block; width: 44px; height: 24px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #cbd5e0; border-radius: 24px; transition: 0.3s; }
        .slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: white; border-radius: 50%; transition: 0.3s; }
        .switch input:checked + .slider { background-color: #3182ce; }
        .switch input:checked + .slider:before { transform: translateX(20px); }
        .switch input:focus + .slider { box-shadow: 0 0 0 3px rgba(49,130,206,0.1); }
        input[readonly] { background: #f7fafc; color: #4a5568; }
        
        /* Custom framework disabled state */
        .custom-disabled { opacity: 0.5; pointer-events: none; background: #f7fafc !important; }
        .section.hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1 style="margin-bottom: 20px;">FederNet Emulation Configuration</h1>
        
        <div class="section">
            <div class="section-title" style="display: flex; justify-content: space-between; align-items: center;">
                <span>Experiment Settings</span>
                <label class="switch-label">
                    <span style="margin-right: 10px;">Custom Framework</span>
                    <div class="switch">
                        <input type="checkbox" id="custom_framework" onchange="toggleCustomFramework()">
                        <span class="slider"></span>
                    </div>
                </label>
            </div>
            <div class="grid-2">
                <div class="form-group">
                    <label>Experiment Name</label>
                    <input type="text" id="experiment_name" value="fl_experiment">
                </div>
                <div class="form-group" id="default_image_group">
                    <label>Docker Image</label>
                    <input type="text" id="default_image" value="fed_opt">
                    <div class="help-text">Docker image to use for all clients if per host docker is empty. To edit enable Custom Framework</div>
                </div>
            </div>
            <div class="grid-3">
                <div class="form-group">
                    <label>Host Single Core Score</label>
                    <input type="text" id="host_single_core_score" value="1079">
                    <div class="help-text">View scores at <a href="https://browser.geekbench.com/" target="_blank" rel="noopener noreferrer">Geekbench Browser</a></div>
                </div>
                <div class="form-group">
                    <label>Enable MQTT Broker</label>
                    <select id="enable_mqtt">
                        <option value="true">Yes</option>
                        <option value="false" selected>No</option>
                    </select>
                 <div class="help-text">Mosquitto required in the Parameter Server Container</div>
                 </div>
                <div class="form-group">
                    <label>Enable Traffic Capture</label>
                    <select id="enable_tcpdump">
                        <option value="true">Yes</option>
                        <option value="false" selected>No</option>
                    </select>
                <div class="help-text">Capture network traffic using tcpdump, tcpdump required in container</div>    
                </div>
            </div>
        </div>

        <div class="section" id="framework_section">
            <div class="section-title">Framework Configuration</div>
            <div class="grid-3">
                <div class="form-group">
                    <label>FL Method</label>
                    <select id="fl_method">
                        <option value="FedAvgN">FedAvgN</option>
                        <option value="FedAvg">FedAvg</option>
                        <option value="FedProx">FedProx</option>
                        <option value="SCAFFOLD">SCAFFOLD</option>
                        <option value="ASOFed">ASOFed</option>
                        <option value="FedAsync">FedAsync</option>
                        <option value="AsyncFedED">AsyncFedED</option>
                        <option value="Unweighted">Unweighted</option>
                        
                    </select>
                </div>
                <div class="form-group">
                    <label>Alpha</label>
                    <input type="number" id="alpha" value="10" step="0.1">
                </div>
                <div class="form-group">
                    <label>Dataset</label>
                    <select id="dataset">
                        <option value="MedMNIST">MedMNIST</option>
                        <option value="mnist">MNIST</option>
                        <option value="cifar10">CIFAR-10</option>
                    </select>
                </div>
            </div>
            <div class="grid-3">
                <div class="form-group">
                    <label>Rounds</label>
                    <input type="number" id="rounds" value="10">
                </div>
                
                <div class="form-group">
                    <label>Min Clients to Start</label>
                    <input type="number" id="min_client_to_start" value="3">
                </div>
                
                <div class="form-group">
                    <label>Clients per Round</label>
                    <input type="number" id="client_round" value="3">
                </div>
                <div class="form-group">
                    <label>Client Epochs</label>
                    <input type="number" id="client_epochs" value="1">
                </div>

                    <div class="form-group">
                    <label>Communication Protocol</label>
                        <select id="protocol">
                            <option value="grpc">gRPC</option>
                            <option value="mqtt">MQTT</option>
                            <option value="websocket">WebSocket</option>
                        </select>
                
                </div>
                    <label class="switch-label" style="justify-self: end;">
                    <span style="margin-right: 10px;">Client Selection</span>
                    <div class="switch">
                        <input type="checkbox" id="client_selection">
                        <span class="slider"></span>
                    </div>
                </label>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Server Configuration</div>
            <div class="client-card">
                <div class="client-header">
                    <span class="client-title">Parameter Server</span>
                    <span class="badge badge-server">SERVER</span>
                </div>

                <!-- Preset selectors -->
                <div class="grid-2" style="margin-bottom: 16px;">
                    <div class="form-group">
                        <label>Device Preset</label>
                        <select id="server_device_preset" onchange="updatePreset('server', 'device')">
                            <option value="">Custom Configuration</option>
                            <option value="rpi4">Raspberry Pi 4</option>
                            <option value="rpi5">Raspberry Pi 5</option>
                            <option value="jetson_nano">NVIDIA Jetson Nano</option>
                            <option value="intel_nuc8">Intel NUC8</option>
                            <option value="smartphone_generic">Generic Smartphone</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Link Preset</label>
                        <select id="server_link_preset" onchange="updatePreset('server', 'link')">
                            <option value="">Custom Configuration</option>
                            <option value="5g">5G Network</option>
                            <option value="wifi5">WiFi 5 (802.11ac)</option>
                            <option value="wifi6">WiFi 6 (802.11ax)</option>
                            <option value="ethernet_1g">1 Gbps Ethernet</option>
                            <option value="ethernet_10g">10 Gbps Ethernet</option>
                        </select>
                    </div>
                </div>

                <!-- Device section -->
                <div class="grid-3">
                    <div class="form-group">
                        <label>CPU Cores</label>
                        <input type="number" id="server_cores" value="4" step="0.5">
                    </div>
                    <div class="form-group">
                        <label>RAM (GB)</label>
                        <input type="number" id="server_ram" value="4" step="0.5">
                    </div>
                    <div class="form-group">
                        <label>Score</label>
                        <input type="number" id="server_score" value="516">
                    </div>
                </div>

                <!-- Link section -->
                <div class="grid-4">
                    <div class="form-group">
                        <label>Bandwidth (Mbps)</label>
                        <input type="number" id="server_bandwidth" value="100">
                    </div>
                    <div class="form-group">
                        <label>Delay (ms)</label>
                        <input type="number" id="server_delay" value="12" step="0.1">
                    </div>
                    <div class="form-group">
                        <label>Jitter (ms)</label>
                        <input type="number" id="server_jitter" value="2" step="0.1">
                    </div>
                    <div class="form-group">
                        <label>Loss (%)</label>
                        <input type="number" id="server_loss" value="0" step="0.1">
                    </div>
                </div>

                <div class="form-group" id="server_image_group">
                    <label>Docker Image (optional)</label>
                    <input type="text" id="server_image" placeholder="Leave empty for default">
                </div>
                <div class="form-group" id="server_command_group">
                    <label>Custom Command (optional)</label>
                    <textarea id="server_command" rows="2" placeholder="python3 custom_server.py --arg value --server_ip {server_address}"></textarea>
                    <div class="help-text">Leave empty to use default command, use {server_address} to get the IP address of the Parameter Server, use {client_address} to get the IP address of the client you are configuring, {client_id} to get the client ID. The PS ID is equal to 0</div>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Client Configuration</div>
            <div class="form-group">
                <label>Number of Clients</label>
                <input type="number" id="num_clients" value="3" min="1">
            </div>
            <button class="btn btn-primary" onclick="generateClients()">Generate Client Cards</button>
            <div class="clients-container" id="clients_container" style="margin-top: 16px;">
                <p style="text-align:center; color: #718096;">Click "Generate Client Cards" to create individual client configurations</p>
            </div>
        </div>

        <div style="margin-top: 24px; display: flex; gap: 12px; justify-content: flex-end;">
            <button class="btn btn-secondary" onclick="saveConfig()">Save Configuration</button>
            <button class="btn btn-primary" onclick="saveAndRun()">Save & Generate YAML</button>
        </div>
    </div>

    <div id="notification" class="notification">
        <span id="notification_text"></span>
    </div>

    <script>
        // Extended device and link presets
        const devicePresets = {
            rpi4: { cores: 4, ram: 4, score: 187 },
            rpi5: { cores: 4, ram: 4, score: 516 },
            jetson_nano: { cores: 6, ram: 4, score: 629 },
            intel_nuc8: { cores: 4, ram: 8, score: 1200 },
            smartphone_generic: { cores: 8, ram: 6, score: 400 }
        };

        const linkPresets = {
            '5g': { bandwidth: 300, delay: 20, jitter: 5, loss: 0.2 },
            'wifi5': { bandwidth: 200, delay: 10, jitter: 3, loss: 0.1 },
            'wifi6': { bandwidth: 600, delay: 8, jitter: 2, loss: 0.05 },
            'ethernet_1g': { bandwidth: 1000, delay: 2, jitter: 0.5, loss: 0.01 },
            'ethernet_10g': { bandwidth: 10000, delay: 1, jitter: 0.2, loss: 0.005 },
            fiber_1gbps: { bandwidth: 1000, delay: 2, jitter: 0.5, loss: 0.01 },
            wifi_80211ac: { bandwidth: 200, delay: 10, jitter: 3, loss: 0.1 },
            '4g_lte': { bandwidth: 25, delay: 45, jitter: 15, loss: 0.5 }
        };

        function toggleCustomFramework() {
            const isCustom = document.getElementById('custom_framework').checked;
            const frameworkSection = document.getElementById('framework_section');
            const defaultImageGroup = document.getElementById('default_image_group');
            const serverImageGroup = document.getElementById('server_image_group');
            const serverCommandGroup = document.getElementById('server_command_group');
            
            if (isCustom) {
                // Hide framework section completely
                frameworkSection.classList.add('hidden');
                
                // Show and enable custom fields
                defaultImageGroup.classList.remove('custom-disabled');
                serverImageGroup.classList.remove('custom-disabled');
                serverCommandGroup.classList.remove('custom-disabled');
                document.getElementById('default_image').disabled = false;
                document.getElementById('server_image').disabled = false;
                document.getElementById('server_command').disabled = false;
            } else {
                // Show framework section
                frameworkSection.classList.remove('hidden');
                
                // Disable custom fields
                defaultImageGroup.classList.add('custom-disabled');
                serverImageGroup.classList.add('custom-disabled');
                serverCommandGroup.classList.add('custom-disabled');
                document.getElementById('default_image').disabled = true;
                document.getElementById('server_image').disabled = true;
                document.getElementById('server_command').disabled = true;
            }
            
            // Update client cards if they exist
            updateClientCustomFields(isCustom);
        }

        function updateClientCustomFields(isCustom) {
            const numClients = parseInt(document.getElementById('num_clients').value);
            for (let i = 1; i <= numClients; i++) {
                const imageGroup = document.getElementById(`client_${i}_image_group`);
                const commandGroup = document.getElementById(`client_${i}_command_group`);
                const imageInput = document.getElementById(`client_${i}_image`);
                const commandInput = document.getElementById(`client_${i}_command`);
                
                if (imageGroup && commandGroup && imageInput && commandInput) {
                    if (isCustom) {
                        imageGroup.classList.remove('custom-disabled');
                        commandGroup.classList.remove('custom-disabled');
                        imageInput.disabled = false;
                        commandInput.disabled = false;
                    } else {
                        imageGroup.classList.add('custom-disabled');
                        commandGroup.classList.add('custom-disabled');
                        imageInput.disabled = true;
                        commandInput.disabled = true;
                    }
                }
            }
        }

        function setInputsEditable(baseId, kind, editable) {
            const fields = kind === 'device'
                ? ['cores','ram','score']
                : ['bandwidth','delay','jitter','loss'];
            fields.forEach(f => {
                const el = document.getElementById(`${baseId}_${f}`);
                if (el) el.readOnly = !editable;
            });
        }

        function updatePreset(baseId, kind) {
            const selectId = `${baseId}_${kind}_preset`;
            const selected = document.getElementById(selectId)?.value || '';
            if (!selected) {
                setInputsEditable(baseId, kind, true);
                return;
            }
            const preset = (kind === 'device') ? devicePresets[selected] : linkPresets[selected];
            if (!preset) {
                setInputsEditable(baseId, kind, true);
                return;
            }
            if (kind === 'device') {
                const { cores, ram, score } = preset;
                const c = document.getElementById(`${baseId}_cores`);
                const r = document.getElementById(`${baseId}_ram`);
                const s = document.getElementById(`${baseId}_score`);
                if (c) c.value = cores;
                if (r) r.value = ram;
                if (s) s.value = score;
            } else {
                const { bandwidth, delay, jitter, loss } = preset;
                const b = document.getElementById(`${baseId}_bandwidth`); 
                const d = document.getElementById(`${baseId}_delay`);
                const j = document.getElementById(`${baseId}_jitter`);
                const l = document.getElementById(`${baseId}_loss`);
                if (b) b.value = bandwidth;
                if (d) d.value = delay;
                if (j) j.value = jitter;
                if (l) l.value = loss;
            }
            setInputsEditable(baseId, kind, false);
        }

        function generateClients() {
            const numClients = parseInt(document.getElementById('num_clients').value);
            const container = document.getElementById('clients_container');
            const isCustom = document.getElementById('custom_framework').checked;
            container.innerHTML = '';

            for (let i = 1; i <= numClients; i++) {
                const card = document.createElement('div');
                card.className = 'client-card';
                const customClass = isCustom ? '' : 'custom-disabled';
                const disabledAttr = isCustom ? '' : 'disabled';
                
                card.innerHTML = `
                    <div class="client-header">
                        <span class="client-title">Client ${i}</span>
                        <span class="badge">CLIENT</span>
                    </div>

                    <div class="grid-2" style="margin-bottom: 16px;">
                        <div class="form-group">
                            <label>Device Preset</label>
                            <select id="client_${i}_device_preset" onchange="updatePreset('client_${i}', 'device')">
                                <option value="">Custom Configuration</option>
                                <option value="rpi4">Raspberry Pi 4</option>
                                <option value="rpi5">Raspberry Pi 5</option>
                                <option value="jetson_nano">NVIDIA Jetson Nano</option>
                                <option value="intel_nuc8">Intel NUC8</option>
                                <option value="smartphone_generic">Generic Smartphone</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Link Preset</label>
                            <select id="client_${i}_link_preset" onchange="updatePreset('client_${i}', 'link')">
                                <option value="">Custom Configuration</option>
                                <option value="5g">5G Network</option>
                                <option value="wifi5">WiFi 5 (802.11ac)</option>
                                <option value="wifi6">WiFi 6 (802.11ax)</option>
                                <option value="ethernet_1g">1 Gbps Ethernet</option>
                                <option value="ethernet_10g">10 Gbps Ethernet</option>
                            </select>
                        </div>
                    </div>

                    <div class="grid-3" style="margin-bottom: 12px;">
                        <div class="form-group">
                            <label>CPU Cores</label>
                            <input type="number" id="client_${i}_cores" value="4" step="0.5">
                        </div>
                        <div class="form-group">
                            <label>RAM (GB)</label>
                            <input type="number" id="client_${i}_ram" value="4" step="0.5">
                        </div>
                        <div class="form-group">
                            <label>Score</label>
                            <input type="number" id="client_${i}_score" value="516">
                        </div>
                    </div>
                    <div class="grid-4" style="margin-bottom: 12px;">
                        <div class="form-group">
                            <label>Bandwidth (Mbps)</label>
                            <input type="number" id="client_${i}_bandwidth" value="25">
                        </div>
                        <div class="form-group">
                            <label>Delay (ms)</label>
                            <input type="number" id="client_${i}_delay" value="45" step="0.1">
                        </div>
                        <div class="form-group">
                            <label>Jitter (ms)</label>
                            <input type="number" id="client_${i}_jitter" value="15" step="0.1">
                        </div>
                        <div class="form-group">
                            <label>Loss (%)</label>
                            <input type="number" id="client_${i}_loss" value="0.5" step="0.1">
                        </div>
                    </div>
                    <div class="form-group ${customClass}" id="client_${i}_image_group">
                        <label>Docker Image (optional)</label>
                        <input type="text" id="client_${i}_image" placeholder="Leave empty for default" ${disabledAttr}>
                    </div>
                    <div class="form-group ${customClass}" id="client_${i}_command_group">
                        <label>Custom Command (optional)</label>
                        <textarea id="client_${i}_command" rows="2" placeholder="python3 custom_client.py --arg value --server_ip {server_address}" ${disabledAttr}></textarea>
                        <div class="help-text">Leave empty to use default command, use {server_address} to get the IP address of the Parameter Server, use {client_address} to get the IP address of the client you are configuring, {client_id} to get the client ID. The PS ID is equal to 0</div>
                    </div>
                `;
                container.appendChild(card);
            }
            
            showNotification(`Generated ${numClients} client cards`, 'success');
        }

        async function saveConfig() {
            const numClients = parseInt(document.getElementById('num_clients').value);
            const isCustom = document.getElementById('custom_framework').checked;
            
            const config = {
                experiment_name: document.getElementById('experiment_name').value,
                image_name: document.getElementById('default_image').value,
                host_single_core_score: parseInt(document.getElementById('host_single_core_score').value),
                enable_mqtt: document.getElementById('enable_mqtt').value === 'true',
                enable_tcpdump: document.getElementById('enable_tcpdump').value === 'true',
                custom_framework: isCustom,
                clients: numClients,
                nodes: []
            };

            // Only include FL framework config if not custom
            if (!isCustom) {
                config.fl_method = document.getElementById('fl_method').value;
                config.alpha = parseFloat(document.getElementById('alpha').value);
                config.rounds = parseInt(document.getElementById('rounds').value);
                config.dataset = document.getElementById('dataset').value;
                
                config.server_config = {
                    min_client_to_start: parseInt(document.getElementById('min_client_to_start').value),
                    client_round: parseInt(document.getElementById('client_round').value)
                };
                config.client_config = {
                    epochs: parseInt(document.getElementById('client_epochs').value)
                };
                config.protocol = document.getElementById('protocol').value;
                
                config.client_selection = document.getElementById('client_selection').checked ? true : false;
            }

            // Server node
            const serverNode = {
                id: 0,
                type: 'server',
                device_preset: document.getElementById('server_device_preset').value || null,
                link_preset: document.getElementById('server_link_preset').value || null,
                device: {
                    cores: parseFloat(document.getElementById('server_cores').value),
                    ram_gib: parseFloat(document.getElementById('server_ram').value),
                    single_core_score: parseInt(document.getElementById('server_score').value)
                },
                link: {
                    bandwidth_mbps: parseInt(document.getElementById('server_bandwidth').value),
                    delay_ms: parseFloat(document.getElementById('server_delay').value),
                    jitter_ms: parseFloat(document.getElementById('server_jitter').value),
                    loss_percent: parseFloat(document.getElementById('server_loss').value)
                }
            };
            
            // Only include custom fields if custom framework is enabled
            if (isCustom) {
                serverNode.image = document.getElementById('server_image').value || null;
                serverNode.custom_command = document.getElementById('server_command').value || null;
            }
            
            config.nodes.push(serverNode);

            // Client nodes
            for (let i = 1; i <= numClients; i++) {
                const clientNode = {
                    id: i,
                    type: 'client',
                    device_preset: document.getElementById(`client_${i}_device_preset`).value || null,
                    link_preset: document.getElementById(`client_${i}_link_preset`).value || null,
                    device: {
                        cores: parseFloat(document.getElementById(`client_${i}_cores`).value),
                        ram_gib: parseFloat(document.getElementById(`client_${i}_ram`).value),
                        single_core_score: parseInt(document.getElementById(`client_${i}_score`).value)
                    },
                    link: {
                        bandwidth_mbps: parseInt(document.getElementById(`client_${i}_bandwidth`).value),
                        delay_ms: parseFloat(document.getElementById(`client_${i}_delay`).value),
                        jitter_ms: parseFloat(document.getElementById(`client_${i}_jitter`).value),
                        loss_percent: parseFloat(document.getElementById(`client_${i}_loss`).value)
                    }
                };
                
                // Only include custom fields if custom framework is enabled
                if (isCustom) {
                    clientNode.image = document.getElementById(`client_${i}_image`).value || null;
                    clientNode.custom_command = document.getElementById(`client_${i}_command`).value || null;
                }
                
                config.nodes.push(clientNode);
            }

            try {
                const response = await fetch('/api/save_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                
                const result = await response.json();
                if (result.success) {
                    showNotification(`Configuration saved: ${result.filename}`, 'success');
                    return result.filename;
                } else {
                    showNotification(`Error: ${result.error}`, 'error');
                    return null;
                }
            } catch (error) {
                showNotification(`Error: ${error}`, 'error');
                return null;
            }
        }

        async function saveAndRun() {
            const filename = await saveConfig();
            if (filename) {
                showNotification('Configuration saved successfully!', 'success');
            }
        }

        function showNotification(message, type) {
            const notification = document.getElementById('notification');
            const text = document.getElementById('notification_text');
            text.textContent = message;
            notification.className = `notification ${type} active`;
            setTimeout(() => notification.classList.remove('active'), 5000);
        }
        
        // Initialize state on page load
        document.addEventListener('DOMContentLoaded', function() {
            toggleCustomFramework();
        });
    </script>
</body>
</html>
'''
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/save_config', methods=['POST'])
def save_config():
    try:
        config_data = request.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = config_data.get('experiment_name', 'experiment').replace(' ', '_')
        filename = f"{config_name}_{timestamp}.yaml"
        filepath = CONFIG_DIR / filename

        # Only include non-null/non-empty values
        top_level_candidates = {
            'experiment_name': config_data.get('experiment_name'),
            'clients': config_data.get('clients'),
            'protocol': config_data.get('protocol'),
            'image_name': config_data.get('image_name'),
            'fl_method': config_data.get('fl_method'),
            'alpha': config_data.get('alpha'),
            'rounds': config_data.get('rounds'),
            'client_selection': config_data.get('client_selection'),
            'server_config': config_data.get('server_config'),
            'client_config': config_data.get('client_config'),
            'host_single_core_score': config_data.get('host_single_core_score'),
            'enable_mqtt': config_data.get('enable_mqtt'),
            'enable_tcpdump': config_data.get('enable_tcpdump'),
        }

        yaml_config = {}
        for k, v in top_level_candidates.items():
            if v is not None and not (isinstance(v, str) and v.strip() == ''):
                yaml_config[k] = v

        yaml_config['nodes'] = []

        for node in config_data.get('nodes', []):
            node_config = {
                'id': node['id'],
                'type': node['type'],
                'constraints': {
                    'cpu_cores': node['device']['cores'],
                    'memory_mb': int(node['device']['ram_gib'] * 1024),
                    'single_core_score': node['device']['single_core_score']
                },
                'link': {
                    'bandwidth_mbps': node['link']['bandwidth_mbps'],
                    'delay_ms': node['link']['delay_ms'],
                    'jitter_ms': node['link']['jitter_ms'],
                    'loss_percent': node['link']['loss_percent']
                }
            }

            # Include optional fields only when present
            if node.get('device_preset'):
                node_config['device_preset'] = node['device_preset']
            if node.get('link_preset'):
                node_config['link_preset'] = node['link_preset']
            if node.get('image'):
                node_config['image'] = node['image']
            if node.get('custom_command'):
                node_config['command'] = node['custom_command']

            yaml_config['nodes'].append(node_config)

        with open(filepath, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

        return jsonify({'success': True, 'filename': filename, 'path': str(filepath)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("FL Emulation Configurator")
    print(f"Config directory: {CONFIG_DIR}")
    print(f"Starting server on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)