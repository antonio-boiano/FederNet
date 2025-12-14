#!/usr/bin/env python3
"""
Web-based Configuration UI for Emulation Experiments (New Format - Complete)
Supports all features: volumes, custom roles, per-container overrides, environment, etc.
"""

from flask import Flask, render_template_string, request, jsonify
import yaml
import json
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_DIR = SCRIPT_DIR.parent / "configs"
CONFIG_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>FL Configuration (Complete)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f7fa; padding: 20px; }
        .container { max-width: 1600px; margin: 0 auto; }
        .section { background: white; padding: 24px; margin-bottom: 20px; border-radius: 8px; border: 1px solid #e2e8f0; }
        .section-title { font-size: 18px; font-weight: 600; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 2px solid #e2e8f0; }
        .subsection { margin: 16px 0; padding: 16px; background: #f7fafc; border-radius: 6px; border-left: 3px solid #3182ce; }
        .form-group { margin-bottom: 16px; }
        label { display: block; margin-bottom: 6px; font-weight: 500; font-size: 14px; color: #4a5568; }
        input, select, textarea { width: 100%; padding: 10px; border: 1px solid #cbd5e0; border-radius: 6px; font-size: 14px; }
        input:focus, select:focus, textarea:focus { outline: none; border-color: #3182ce; box-shadow: 0 0 0 3px rgba(49,130,206,0.1); }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
        .container-card { background: #f7fafc; border: 2px solid #3182ce; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
        .role-card { background: #edf2f7; border: 2px solid #805ad5; border-radius: 8px; padding: 20px; margin-bottom: 16px; }
        .card-header { font-weight: 600; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #cbd5e0; display: flex; justify-content: space-between; align-items: center; }
        .badge { background: #3182ce; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 500; display: inline-block; }
        .badge-server { background: #805ad5; }
        .badge-custom { background: #ed8936; }
        .btn { padding: 10px 20px; border: none; border-radius: 6px; font-size: 14px; font-weight: 500; cursor: pointer; transition: all 0.2s; }
        .btn-primary { background: #3182ce; color: white; }
        .btn-primary:hover { background: #2c5282; }
        .btn-secondary { background: #718096; color: white; }
        .btn-secondary:hover { background: #4a5568; }
        .btn-success { background: #48bb78; color: white; }
        .btn-success:hover { background: #38a169; }
        .btn-danger { background: #f56565; color: white; }
        .btn-danger:hover { background: #c53030; }
        .btn-small { padding: 6px 12px; font-size: 12px; }
        .help-text { font-size: 12px; color: #718096; margin-top: 4px; }
        .notification { position: fixed; top: 20px; right: 20px; padding: 14px 20px; background: white; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); transform: translateX(400px); transition: transform 0.3s; z-index: 1000; }
        .notification.active { transform: translateX(0); }
        .notification.success { border-left: 4px solid #48bb78; }
        .notification.error { border-left: 4px solid #f56565; }
        .list-row { display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: end; margin-bottom: 8px; }
        .dict-row { display: grid; grid-template-columns: 1fr 1fr auto; gap: 12px; align-items: end; margin-bottom: 8px; }
        .containers-list { max-height: 600px; overflow-y: auto; }
        .toggle-switch { position: relative; display: inline-block; width: 50px; height: 24px; margin-left: 10px; }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #cbd5e0; border-radius: 24px; transition: 0.3s; }
        .toggle-slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: white; border-radius: 50%; transition: 0.3s; }
        .toggle-switch input:checked + .toggle-slider { background-color: #3182ce; }
        .toggle-switch input:checked + .toggle-slider:before { transform: translateX(26px); }
        .hidden { display: none !important; }
        .mode-toggle { display: flex; align-items: center; justify-content: space-between; padding: 16px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; color: white; margin-bottom: 20px; }
        .mode-toggle h2 { margin: 0; font-size: 24px; }
        .mode-label { display: flex; align-items: center; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <div class="mode-toggle">
            <h2>FederNet Configuration</h2>
            <label class="mode-label">
                Simple Mode
                <div class="toggle-switch">
                    <input type="checkbox" id="advanced_mode" onchange="toggleAdvancedMode()">
                    <span class="toggle-slider"></span>
                </div>
                Advanced Mode
            </label>
        </div>

        <!-- CONTAINERNET SECTION -->
        <div class="section">
            <div class="section-title">Containernet Configuration</div>
            <div class="grid-3">
                <div class="form-group">
                    <label>Number of Containers</label>
                    <input type="number" id="num_containers" value="3" min="1">
                    <div class="help-text">Total containers including server</div>
                </div>
                <div class="form-group">
                    <label>Default Docker Image</label>
                    <input type="text" id="image_name" value="anboiano/fedopt:latest">
                </div>
                <div class="form-group">
                    <label>Host Single Core Score</label>
                    <input type="number" id="host_single_core_score" value="1554">
                </div>
            </div>
            <div class="grid-3">
                <div class="form-group">
                    <label>Device Variance</label>
                    <input type="number" id="device_variance" value="0.05" step="0.01">
                </div>
                <div class="form-group">
                    <label>Enable Traffic Capture</label>
                    <select id="enable_tcpdump">
                        <option value="true">Yes</option>
                        <option value="false" selected>No</option>
                    </select>
                </div>
            </div>

            <!-- Global Volumes (Advanced) -->
            <div id="global_volumes_section" class="subsection hidden">
                <h3 style="margin-bottom: 12px;">Global Volumes (mounted to all containers)</h3>
                <div class="help-text" style="margin-bottom: 12px;">Format: ./host/path:/container/path or ./host/path:/container/path:ro</div>
                <div id="global_volumes_container"></div>
                <button class="btn btn-secondary btn-small" onclick="addGlobalVolume()">Add Volume</button>
            </div>

            <button class="btn btn-success" onclick="generateContainersList()" style="margin-top: 16px;">Generate Container Configuration</button>

            <div id="containers_list" class="containers-list" style="margin-top: 20px;">
                <p style="text-align:center; color: #718096;">Click "Generate Container Configuration"</p>
            </div>
        </div>

        <!-- APPLICATION SECTION -->
        <div class="section">
            <div class="section-title">Application Configuration</div>
            <div class="form-group">
                <label>Application Name</label>
                <input type="text" id="app_name" value="fl_experiment">
            </div>

            <div class="subsection">
                <h3 style="margin-bottom: 12px;">Global Variables</h3>
                <div class="help-text" style="margin-bottom: 12px;">Use {variable_name} in commands for substitution</div>
                <div id="variables_container"></div>
                <button class="btn btn-secondary btn-small" onclick="addVariable()">Add Variable</button>
            </div>
        </div>

        <!-- ROLES SECTION -->
        <div class="section">
            <div class="section-title">Role Configuration</div>

            <!-- Server Role -->
            <div id="server_role" class="role-card">
                <div class="card-header">
                    <span>Server Role <span class="badge badge-server">SERVER</span></span>
                </div>
                <div class="grid-2">
                    <div class="form-group">
                        <label>Container ID(s)</label>
                        <input type="text" id="server_container_ids" value="0">
                        <div class="help-text">Comma-separated IDs or single ID</div>
                    </div>
                    <div class="form-group">
                        <label>Startup Delay (seconds)</label>
                        <input type="number" id="server_startup_delay" value="0" min="0">
                    </div>
                </div>
                <div class="form-group">
                    <label>Command Template</label>
                    <textarea id="server_command" rows="3">python3 -u run.py --protocol {protocol} --mode Server --port {port} --ip {container_ip} --index {container_id} --rounds {rounds} --fl_method {fl_method} --alpha {alpha} --min_clients {min_clients} --num_client {num_client}</textarea>
                </div>
                <div class="form-group">
                    <label>Pre-Commands (one per line)</label>
                    <textarea id="server_pre_commands" rows="2" placeholder="echo 'Starting server'"></textarea>
                </div>
                <div id="server_advanced" class="hidden">
                    <div class="form-group">
                        <label>Post-Commands (one per line)</label>
                        <textarea id="server_post_commands" rows="2" placeholder="echo 'Server completed'"></textarea>
                    </div>
                    <div class="grid-2">
                        <div class="form-group">
                            <label>Docker Image Override</label>
                            <input type="text" id="server_image" placeholder="Leave empty for default">
                        </div>
                        <div class="form-group">
                            <label>Working Directory</label>
                            <input type="text" id="server_working_dir" placeholder="/app">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Environment Variables</label>
                        <div id="server_env_container"></div>
                        <button class="btn btn-secondary btn-small" onclick="addEnvVar('server')">Add Environment Variable</button>
                    </div>
                    <div class="form-group">
                        <label>Role-Specific Volumes</label>
                        <div id="server_volumes_container"></div>
                        <button class="btn btn-secondary btn-small" onclick="addRoleVolume('server')">Add Volume</button>
                    </div>
                </div>
            </div>

            <!-- Client Role -->
            <div id="client_role" class="role-card">
                <div class="card-header">
                    <span>Client Role <span class="badge">CLIENT</span></span>
                </div>
                <div class="grid-2">
                    <div class="form-group">
                        <label>Container Assignment</label>
                        <select id="client_container_mode">
                            <option value="all_except_server">All except server</option>
                            <option value="custom">Custom IDs</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Startup Delay (seconds)</label>
                        <input type="number" id="client_startup_delay" value="10" min="0">
                    </div>
                </div>
                <div class="form-group" id="client_custom_ids_group" style="display: none;">
                    <label>Custom Container IDs (comma-separated)</label>
                    <input type="text" id="client_container_ids" placeholder="1,2,3">
                </div>
                <div class="form-group">
                    <label>Command Template</label>
                    <textarea id="client_command" rows="3">python3 -u run.py --protocol {protocol} --mode Client --my_ip {container_ip} --port {port} --ip {ip_0} --index {container_id} --epochs {epochs} --fl_method {fl_method} --alpha {alpha}</textarea>
                </div>
                <div class="form-group">
                    <label>Pre-Commands (one per line)</label>
                    <textarea id="client_pre_commands" rows="2" placeholder="echo 'Starting client {container_id}'"></textarea>
                </div>
                <div id="client_advanced" class="hidden">
                    <div class="form-group">
                        <label>Post-Commands (one per line)</label>
                        <textarea id="client_post_commands" rows="2" placeholder="echo 'Client completed'"></textarea>
                    </div>
                    <div class="grid-2">
                        <div class="form-group">
                            <label>Docker Image Override</label>
                            <input type="text" id="client_image" placeholder="Leave empty for default">
                        </div>
                        <div class="form-group">
                            <label>Working Directory</label>
                            <input type="text" id="client_working_dir" placeholder="/app">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Environment Variables</label>
                        <div id="client_env_container"></div>
                        <button class="btn btn-secondary btn-small" onclick="addEnvVar('client')">Add Environment Variable</button>
                    </div>
                    <div class="form-group">
                        <label>Role-Specific Volumes</label>
                        <div id="client_volumes_container"></div>
                        <button class="btn btn-secondary btn-small" onclick="addRoleVolume('client')">Add Volume</button>
                    </div>
                </div>
            </div>

            <!-- Custom Roles (Advanced) -->
            <div id="custom_roles_section" class="hidden">
                <div id="custom_roles_container"></div>
                <button class="btn btn-success btn-small" onclick="addCustomRole()">Add Custom Role</button>
            </div>
        </div>

        <div style="margin-top: 24px; display: flex; gap: 12px; justify-content: flex-end;">
            <button class="btn btn-primary" onclick="saveConfig()">Generate & Save YAML</button>
        </div>
    </div>

    <div id="notification" class="notification">
        <span id="notification_text"></span>
    </div>

    <script>
        const deviceTypes = ['none', 'rpi4', 'rpi5', 'jetson_nano', 'intel_nuc8', 'smartphone_generic'];
        const networkTypes = ['none', 'wifi_80211ac', '4g_lte', '4g_lte_advanced', '5g_sub6', '5g_mmwave', 'satellite_leo_starlink', 'ethernet_1g', 'ethernet_10g'];
        let customRoleCount = 0;

        // Toggle advanced mode
        function toggleAdvancedMode() {
            const isAdvanced = document.getElementById('advanced_mode').checked;
            // Show/hide advanced sections
            document.getElementById('global_volumes_section').classList.toggle('hidden', !isAdvanced);
            document.getElementById('server_advanced').classList.toggle('hidden', !isAdvanced);
            document.getElementById('client_advanced').classList.toggle('hidden', !isAdvanced);
            document.getElementById('custom_roles_section').classList.toggle('hidden', !isAdvanced);
        }

        // Client container mode
        document.getElementById('client_container_mode').addEventListener('change', function() {
            document.getElementById('client_custom_ids_group').style.display =
                this.value === 'custom' ? 'block' : 'none';
        });

        // Generate containers list
        function generateContainersList() {
            const numContainers = parseInt(document.getElementById('num_containers').value);
            const container = document.getElementById('containers_list');
            container.innerHTML = '';

            for (let i = 0; i < numContainers; i++) {
                const isServer = i === 0;
                const card = document.createElement('div');
                card.className = 'container-card';
                card.innerHTML = `
                    <div class="card-header">
                        <span>Container ${i} <span class="badge ${isServer ? 'badge-server' : ''}">${isServer ? 'SERVER' : 'CLIENT'}</span></span>
                    </div>
                    <div class="grid-2">
                        <div class="form-group">
                            <label>Device Type</label>
                            <select id="device_type_${i}">
                                ${deviceTypes.map(dt => `<option value="${dt}"${dt === (isServer ? 'none' : 'intel_nuc8') ? ' selected' : ''}>${dt}</option>`).join('')}
                            </select>
                            <div class="help-text">'none' = no CPU/memory constraints</div>
                        </div>
                        <div class="form-group">
                            <label>Network Type</label>
                            <select id="network_type_${i}">
                                ${networkTypes.map(nt => `<option value="${nt}"${nt === (isServer ? 'none' : '4g_lte') ? ' selected' : ''}>${nt}</option>`).join('')}
                            </select>
                            <div class="help-text">'none' = unlimited bandwidth, zero delay</div>
                        </div>
                    </div>
                `;
                container.appendChild(card);
            }
            showNotification(`Generated ${numContainers} containers`, 'success');
        }

        // Variables
        function addVariable() {
            const container = document.getElementById('variables_container');
            const row = document.createElement('div');
            row.className = 'dict-row';
            row.innerHTML = `
                <input type="text" placeholder="Variable name" class="var-name">
                <input type="text" placeholder="Value" class="var-value">
                <button class="btn btn-danger btn-small" onclick="this.parentElement.remove()">Remove</button>
            `;
            container.appendChild(row);
        }

        // Global volumes
        function addGlobalVolume() {
            const container = document.getElementById('global_volumes_container');
            const row = document.createElement('div');
            row.className = 'list-row';
            row.innerHTML = `
                <input type="text" placeholder="./host/path:/container/path" class="volume-spec">
                <button class="btn btn-danger btn-small" onclick="this.parentElement.remove()">Remove</button>
            `;
            container.appendChild(row);
        }

        // Environment variables
        function addEnvVar(role) {
            const container = document.getElementById(`${role}_env_container`);
            const row = document.createElement('div');
            row.className = 'dict-row';
            row.innerHTML = `
                <input type="text" placeholder="KEY" class="env-key">
                <input type="text" placeholder="value" class="env-value">
                <button class="btn btn-danger btn-small" onclick="this.parentElement.remove()">Remove</button>
            `;
            container.appendChild(row);
        }

        // Role volumes
        function addRoleVolume(role) {
            const container = document.getElementById(`${role}_volumes_container`);
            const row = document.createElement('div');
            row.className = 'list-row';
            row.innerHTML = `
                <input type="text" placeholder="./path:/container/path" class="role-volume">
                <button class="btn btn-danger btn-small" onclick="this.parentElement.remove()">Remove</button>
            `;
            container.appendChild(row);
        }

        // Custom roles
        function addCustomRole() {
            const roleId = `custom_role_${customRoleCount++}`;
            const container = document.getElementById('custom_roles_container');
            const card = document.createElement('div');
            card.className = 'role-card';
            card.id = roleId;
            card.innerHTML = `
                <div class="card-header">
                    <div>
                        <input type="text" placeholder="Role name" id="${roleId}_name" style="display: inline-block; width: 200px; margin-right: 10px;">
                        <span class="badge badge-custom">CUSTOM</span>
                    </div>
                    <button class="btn btn-danger btn-small" onclick="document.getElementById('${roleId}').remove()">Remove Role</button>
                </div>
                <div class="grid-2">
                    <div class="form-group">
                        <label>Container IDs (comma-separated)</label>
                        <input type="text" id="${roleId}_container_ids" placeholder="3,4,5">
                    </div>
                    <div class="form-group">
                        <label>Startup Delay (seconds)</label>
                        <input type="number" id="${roleId}_startup_delay" value="0" min="0">
                    </div>
                </div>
                <div class="form-group">
                    <label>Command Template</label>
                    <textarea id="${roleId}_command" rows="3" placeholder="your command here"></textarea>
                </div>
                <div class="form-group">
                    <label>Pre-Commands</label>
                    <textarea id="${roleId}_pre_commands" rows="2"></textarea>
                </div>
                <div class="grid-2">
                    <div class="form-group">
                        <label>Docker Image</label>
                        <input type="text" id="${roleId}_image" placeholder="Optional">
                    </div>
                    <div class="form-group">
                        <label>Working Directory</label>
                        <input type="text" id="${roleId}_working_dir" placeholder="/app">
                    </div>
                </div>
            `;
            container.appendChild(card);
        }

        // Save configuration
        async function saveConfig() {
            const numContainers = parseInt(document.getElementById('num_containers').value);

            // Collect device/network types
            const deviceTypes = [], networkTypes = [];
            for (let i = 0; i < numContainers; i++) {
                const devSel = document.getElementById(`device_type_${i}`);
                const netSel = document.getElementById(`network_type_${i}`);
                if (!devSel || !netSel) {
                    showNotification('Please generate container configuration first', 'error');
                    return;
                }
                deviceTypes.push(devSel.value);
                networkTypes.push(netSel.value);
            }

            // Collect variables
            const variables = {};
            document.querySelectorAll('#variables_container .dict-row').forEach(row => {
                const name = row.querySelector('.var-name').value.trim();
                const value = row.querySelector('.var-value').value.trim();
                if (name) {
                    variables[name] = isNaN(value) ? (value === 'true' ? true : value === 'false' ? false : value) : parseFloat(value);
                }
            });

            // Collect global volumes
            const globalVolumes = [];
            document.querySelectorAll('#global_volumes_container .list-row').forEach(row => {
                const vol = row.querySelector('.volume-spec').value.trim();
                if (vol) globalVolumes.push(vol);
            });

            // Helper function to parse comma-separated IDs
            function parseContainerIds(str) {
                if (str === 'all_except_server') return str;
                return str.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
            }

            // Helper to collect env vars
            function collectEnvVars(containerId) {
                const env = {};
                document.querySelectorAll(`#${containerId} .dict-row`).forEach(row => {
                    const key = row.querySelector('.env-key')?.value.trim();
                    const val = row.querySelector('.env-value')?.value.trim();
                    if (key) env[key] = val;
                });
                return env;
            }

            // Helper to collect volumes
            function collectVolumes(containerId) {
                const vols = [];
                document.querySelectorAll(`#${containerId} .list-row`).forEach(row => {
                    const vol = row.querySelector('.role-volume')?.value.trim();
                    if (vol) vols.push(vol);
                });
                return vols;
            }

            // Helper to split lines
            function splitLines(text) {
                return text.split('\\n').map(s => s.trim()).filter(s => s);
            }

            // Build roles
            const roles = {};
            const roleOrder = [];

            // Server role
            const serverIds = document.getElementById('server_container_ids').value.trim();
            roles.server = {
                container_ids: parseContainerIds(serverIds),
                command: document.getElementById('server_command').value.trim(),
                startup_delay: parseInt(document.getElementById('server_startup_delay').value),
                wait_for_completion: true
            };
            const serverPre = splitLines(document.getElementById('server_pre_commands').value);
            if (serverPre.length > 0) roles.server.pre_commands = serverPre;

            if (document.getElementById('advanced_mode').checked) {
                const serverPost = splitLines(document.getElementById('server_post_commands').value);
                if (serverPost.length > 0) roles.server.post_commands = serverPost;

                const serverImg = document.getElementById('server_image').value.trim();
                if (serverImg) roles.server.image = serverImg;

                const serverWd = document.getElementById('server_working_dir').value.trim();
                if (serverWd) roles.server.working_dir = serverWd;

                const serverEnv = collectEnvVars('server_env_container');
                if (Object.keys(serverEnv).length > 0) roles.server.environment = serverEnv;

                const serverVols = collectVolumes('server_volumes_container');
                if (serverVols.length > 0) roles.server.volumes = serverVols;
            }
            roleOrder.push('server');

            // Client role
            const clientMode = document.getElementById('client_container_mode').value;
            const clientIds = clientMode === 'all_except_server' ?
                'all_except_server' :
                parseContainerIds(document.getElementById('client_container_ids').value);

            roles.client = {
                container_ids: clientIds,
                command: document.getElementById('client_command').value.trim(),
                startup_delay: parseInt(document.getElementById('client_startup_delay').value),
                wait_for_completion: true
            };
            const clientPre = splitLines(document.getElementById('client_pre_commands').value);
            if (clientPre.length > 0) roles.client.pre_commands = clientPre;

            if (document.getElementById('advanced_mode').checked) {
                const clientPost = splitLines(document.getElementById('client_post_commands').value);
                if (clientPost.length > 0) roles.client.post_commands = clientPost;

                const clientImg = document.getElementById('client_image').value.trim();
                if (clientImg) roles.client.image = clientImg;

                const clientWd = document.getElementById('client_working_dir').value.trim();
                if (clientWd) roles.client.working_dir = clientWd;

                const clientEnv = collectEnvVars('client_env_container');
                if (Object.keys(clientEnv).length > 0) roles.client.environment = clientEnv;

                const clientVols = collectVolumes('client_volumes_container');
                if (clientVols.length > 0) roles.client.volumes = clientVols;
            }
            roleOrder.push('client');

            // Custom roles (if any)
            document.querySelectorAll('#custom_roles_container .role-card').forEach(card => {
                const roleId = card.id;
                const roleName = document.getElementById(`${roleId}_name`).value.trim();
                if (!roleName) return;

                const ids = parseContainerIds(document.getElementById(`${roleId}_container_ids`).value);
                roles[roleName] = {
                    container_ids: ids,
                    command: document.getElementById(`${roleId}_command`).value.trim(),
                    startup_delay: parseInt(document.getElementById(`${roleId}_startup_delay`).value),
                    wait_for_completion: true
                };

                const pre = splitLines(document.getElementById(`${roleId}_pre_commands`).value);
                if (pre.length > 0) roles[roleName].pre_commands = pre;

                const img = document.getElementById(`${roleId}_image`).value.trim();
                if (img) roles[roleName].image = img;

                const wd = document.getElementById(`${roleId}_working_dir`).value.trim();
                if (wd) roles[roleName].working_dir = wd;

                roleOrder.push(roleName);
            });

            // Build config
            const config = {
                containernet: {
                    num_containers: numContainers,
                    image_name: document.getElementById('image_name').value,
                    device_type: deviceTypes,
                    network_type: networkTypes,
                    host_single_core_score: parseInt(document.getElementById('host_single_core_score').value),
                    device_variance: parseFloat(document.getElementById('device_variance').value),
                    enable_tcpdump: document.getElementById('enable_tcpdump').value === 'true'
                },
                application: {
                    name: document.getElementById('app_name').value,
                    variables: variables,
                    roles: roles,
                    role_order: roleOrder
                }
            };

            // Add global volumes if any
            if (globalVolumes.length > 0) {
                config.containernet.volumes = globalVolumes;
            }

            try {
                const response = await fetch('/api/save_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                const result = await response.json();
                if (result.success) {
                    showNotification(`Saved: ${result.filename}`, 'success');
                } else {
                    showNotification(`Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showNotification(`Error: ${error}`, 'error');
            }
        }

        function showNotification(message, type) {
            const notification = document.getElementById('notification');
            const text = document.getElementById('notification_text');
            text.textContent = message;
            notification.className = `notification ${type} active`;
            setTimeout(() => notification.classList.remove('active'), 5000);
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            // Add default variables
            const defaultVars = [
                {name: 'port', value: '1883'},
                {name: 'fl_method', value: 'FedAvgN'},
                {name: 'alpha', value: '10'},
                {name: 'rounds', value: '2'},
                {name: 'epochs', value: '5'},
                {name: 'min_clients', value: '2'},
                {name: 'num_client', value: '2'}
            ];
            defaultVars.forEach(v => {
                const row = document.createElement('div');
                row.className = 'dict-row';
                row.innerHTML = `
                    <input type="text" placeholder="Variable name" class="var-name" value="${v.name}">
                    <input type="text" placeholder="Value" class="var-value" value="${v.value}">
                    <button class="btn btn-danger btn-small" onclick="this.parentElement.remove()">Remove</button>
                `;
                document.getElementById('variables_container').appendChild(row);
            });
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
        app_name = config_data.get('application', {}).get('name', 'experiment').replace(' ', '_')
        filename = f"{app_name}_{timestamp}.yaml"
        filepath = CONFIG_DIR / filename

        with open(filepath, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        return jsonify({'success': True, 'filename': filename, 'path': str(filepath)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("FederNet Configuration UI (Complete)")
    print(f"Config directory: {CONFIG_DIR}")
    print(f"Starting server on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
