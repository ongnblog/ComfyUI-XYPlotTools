import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_CONFIG = {
    OGN_XYPlot: {
        cellProgress: true,
    },
    OGN_XYCheckpointAxis: {
        label: "Checkpoint",
        button: "+ Add Checkpoint",
        rowRemoveButton: "- Remove",
        rows: [{ kind: "combo", prefix: "checkpoint_", source: "checkpoint_1" }],
    },
    OGN_XYDiffusionModelAxis: {
        label: "Diffusion Model",
        button: "+ Add Diffusion Model",
        rowRemoveButton: "- Remove",
        rows: [{ kind: "combo", prefix: "diffusion_model_", source: "diffusion_model_1" }],
    },
    OGN_XYSamplerAxis: {
        label: "Sampler",
        button: "+ Add Sampler",
        rowRemoveButton: "- Remove",
        rows: [{ kind: "combo", prefix: "sampler_", source: "sampler_1" }],
    },
    OGN_XYLoraAxis: {
        label: "LoRA",
        loraSets: true,
    },
    OGN_XYLoraEpochRangeAxis: {
        label: "LoRA Epoch Range",
        loraEpochRange: true,
    },
    OGN_XYPromptSRAxis: {
        label: "Prompt S/R",
        button: "+ Add Replacement",
        removeButton: "- Remove Replacement",
        rows: [{ kind: "string", prefix: "replace_", value: "", multiline: true }],
    },
};

function getOptions(nodeData, source) {
    const required = nodeData?.input?.required || {};
    const spec = required[source];
    if (Array.isArray(spec?.[0])) {
        return spec[0];
    }
    return ["None"];
}

function getRegisteredExtensions() {
  const candidates = [
    app.extensions,
    app.extensionManager?.extensions,
    app.extensionManager?.registeredExtensions,
    app.extensionManager?.registrations,
  ];
  for (const candidate of candidates) {
    if (Array.isArray(candidate)) {
      return candidate;
    }
    if (candidate instanceof Map) {
      return [...candidate.values()];
    }
  }
  return [];
}

function refreshModelManagerWidgets(node) {
  const extension = getRegisteredExtensions().find(
    (item) => item?.name === "ComfyUI.ModelManager.ThumbnailTooltips"
  );
  extension?.nodeCreated?.(node);
}

function nextIndex(node, config) {
    let index = 1;
    for (const widget of node.widgets || []) {
        for (const row of config.rows) {
            if (widget.name?.startsWith(row.prefix)) {
                const value = Number(widget.name.slice(row.prefix.length));
                if (!Number.isNaN(value)) {
                    index = Math.max(index, value);
                }
            }
        }
    }
    return index + 1;
}

function moveButtonToEnd(node) {
    if (!node.widgets) {
        return;
    }
    if (node.ognAddButton && !node.ognButtonSpacer) {
        node.ognButtonSpacer = {
            type: "custom",
            name: "ogn_button_spacer",
            value: null,
            serialize: false,
            computeSize: () => [0, 10],
            draw: () => { },
        };
    }
    if (node.ognButtonSpacer) {
        const spacerIndex = node.widgets.indexOf(node.ognButtonSpacer);
        if (spacerIndex >= 0) {
            node.widgets.splice(spacerIndex, 1);
        }
        node.widgets.push(node.ognButtonSpacer);
    }
    for (const button of [node.ognAddButton, node.ognRemoveButton]) {
        if (!button) continue;
        const index = node.widgets.indexOf(button);
        if (index >= 0) {
            node.widgets.splice(index, 1);
        }
        node.widgets.push(button);
    }
}

function addWidget(node, row, index, nodeData, value) {
    const name = `${row.prefix}${index}`;
    if ((node.widgets || []).some((widget) => widget.name === name)) {
        return;
    }
    if (row.kind === "combo") {
        const options = getOptions(nodeData, row.source);
        const selected = options.includes(value) ? value : (value ?? options[0] ?? "None");
        const widget = node.addWidget("combo", name, selected, null, { values: options });
        refreshModelManagerWidgets(node);
        return widget;
    } else if (row.kind === "float") {
        return node.addWidget("number", name, value ?? row.value ?? 1.0, null, {
            min: -20,
            max: 20,
            step: 0.05,
            precision: 3,
        });
    } else {
        if (row.multiline === true && ComfyWidgets?.STRING) {
            const widget = ComfyWidgets.STRING(
                node,
                name,
                ["STRING", { multiline: true, default: value ?? row.value ?? "" }],
                app
            ).widget;
            widget.value = value ?? row.value ?? "";
            return;
        }
        return node.addWidget("text", name, value ?? row.value ?? "", null, {});
    }
}

function removeRow(node, config, rowIndex) {
    for (const widget of [...(node.widgets || [])]) {
        const isRowWidget = config.rows.some((row) => widget.name === `${row.prefix}${rowIndex}`);
        const isRemoveButton = widget.ognRowRemoveIndex === rowIndex;
        const isSpacer = widget.ognRowSpacerIndex === rowIndex;
        if (!isRowWidget && !isRemoveButton && !isSpacer) {
            continue;
        }
        const widgetIndex = node.widgets.indexOf(widget);
        if (widgetIndex >= 0) {
            widget.onRemove?.();
            node.widgets.splice(widgetIndex, 1);
        }
    }
    resizeNode(node);
}

function addRowRemoveButton(node, config, index) {
    if (!config.rowRemoveButton || index <= 1) {
        return;
    }
    if ((node.widgets || []).some((widget) => widget.ognRowRemoveIndex === index)) {
        return;
    }
    const widget = node.addWidget("button", config.rowRemoveButton, null, () => {
        removeRow(node, config, index);
    });
    widget.ognRowRemoveIndex = index;
}

function addRowSpacer(node, config, index) {
    if (!config.rowRemoveButton || index <= 1) {
        return;
    }
    if ((node.widgets || []).some((widget) => widget.ognRowSpacerIndex === index)) {
        return;
    }
    const widget = {
        type: "custom",
        name: `ogn_spacer_${config.label}_${index}`,
        value: null,
        serialize: false,
        ognRowSpacerIndex: index,
        computeSize: () => [0, 10],
        draw: () => { },
    };
    node.widgets.push(widget);
}

function addRow(node, config, nodeData, values = null) {
    const index = nextIndex(node, config);
    addRowSpacer(node, config, index);
    config.rows.forEach((row, rowIndex) => {
        addWidget(node, row, index, nodeData, values?.[rowIndex]);
        if (rowIndex === 0) {
            addRowRemoveButton(node, config, index);
        }
    });
    moveButtonToEnd(node);
    resizeNode(node);
}

function removeLastRow(node, config) {
    const widgets = dynamicWidgets(node, config);
    if (!widgets.length) {
        return;
    }
    let lastIndex = 1;
    for (const widget of widgets) {
        for (const row of config.rows) {
            if (!widget.name?.startsWith(row.prefix)) continue;
            const index = Number(widget.name.slice(row.prefix.length));
            if (Number.isFinite(index)) {
                lastIndex = Math.max(lastIndex, index);
            }
        }
    }
    for (const widget of [...widgets]) {
        for (const row of config.rows) {
            if (!widget.name?.startsWith(row.prefix)) continue;
            const index = Number(widget.name.slice(row.prefix.length));
            if (index === lastIndex) {
                const widgetIndex = node.widgets.indexOf(widget);
                if (widgetIndex >= 0) {
                    widget.onRemove?.();
                    node.widgets.splice(widgetIndex, 1);
                }
            }
        }
    }
    resizeNode(node);
}

function resizeNode(node) {
    if (!node.computeSize) {
        return;
    }
    const size = node.computeSize();
    node.setSize?.([
        Math.max(node.size[0], size[0]),
        Math.max(node.size[1], size[1] + 8),
    ]);
    node.setDirtyCanvas(true, true);
}

function queueNodeResize(node) {
    requestAnimationFrame(() => {
        resizeNode(node);
        requestAnimationFrame(() => resizeNode(node));
    });
}

function dynamicWidgets(node, config) {
    const prefixes = config.rows.map((row) => row.prefix);
    return (node.widgets || []).filter((widget) => {
        if (widget === node.ognAddButton) {
            return false;
        }
        if (widget === node.ognRemoveButton) {
            return false;
        }
        if (widget.ognRowRemoveIndex) {
            return false;
        }
        if (widget.ognRowSpacerIndex) {
            return false;
        }
        const prefix = prefixes.find((candidate) => widget.name?.startsWith(candidate));
        if (!prefix) {
            return false;
        }
        const index = Number(widget.name.slice(prefix.length));
        return Number.isFinite(index) && index > 1;
    });
}

function saveDynamicWidgets(node, config) {
    return dynamicWidgets(node, config).map((widget) => ({
        name: widget.name,
        value: widget.value,
    }));
}

function restoreDynamicWidgets(node, config, nodeData, saved) {
    if (!Array.isArray(saved) || !saved.length) {
        return;
    }
    const rowsByIndex = new Map();
    for (const item of saved) {
        const row = config.rows.find((candidate) => item.name?.startsWith(candidate.prefix));
        if (!row) {
            continue;
        }
        const index = Number(item.name.slice(row.prefix.length));
        if (!Number.isFinite(index) || index <= 1) {
            continue;
        }
        const values = rowsByIndex.get(index) ?? [];
        values[config.rows.indexOf(row)] = item.value;
        rowsByIndex.set(index, values);
    }
    for (const [index, values] of [...rowsByIndex.entries()].sort((a, b) => a[0] - b[0])) {
        addRowSpacer(node, config, index);
        config.rows.forEach((row, rowIndex) => {
            addWidget(node, row, index, nodeData, values?.[rowIndex]);
            if (rowIndex === 0) {
                addRowRemoveButton(node, config, index);
            }
        });
    }
    moveButtonToEnd(node);
    resizeNode(node);
}

function parseLoraWidgetName(name) {
    let match = /^lora_(\d+)$/.exec(name ?? "");
    if (match) return { kind: "lora", index: Number(match[1]) };
    match = /^strength_model_(\d+)$/.exec(name ?? "");
    if (match) return { kind: "strength", index: Number(match[1]) };
    match = /^group_number_(\d+)$/.exec(name ?? "");
    if (match) return { kind: "group", index: Number(match[1]) };

    match = /^lora(\d+)_(\d+)$/.exec(name ?? "");
    if (match) return { kind: "lora", index: Number(match[1]) * 10000 + Number(match[2]), group: Number(match[1]) };
    match = /^strength_model_(\d+)_(\d+)$/.exec(name ?? "");
    if (match) return { kind: "strength", index: Number(match[1]) * 10000 + Number(match[2]), group: Number(match[1]) };
    match = /^group_number_(\d+)_(\d+)$/.exec(name ?? "");
    if (match) return { kind: "group", index: Number(match[1]) * 10000 + Number(match[2]), group: Number(match[1]) };
    return null;
}

function loraWidgetName(kind, index) {
    if (kind === "lora") return `lora_${index}`;
    if (kind === "strength") return `strength_model_${index}`;
    return `group_number_${index}`;
}

function loraRowIndexes(node) {
    const indexes = new Set([1]);
    for (const widget of node.widgets || []) {
        const parsed = parseLoraWidgetName(widget.name);
        if (parsed) indexes.add(parsed.index);
    }
    return [...indexes].sort((a, b) => a - b);
}

function nextLoraRowIndex(node) {
    return Math.max(...loraRowIndexes(node)) + 1;
}

function addLoraValueWidget(node, nodeData, index, kind, value = null) {
    const name = loraWidgetName(kind, index);
    if ((node.widgets || []).some((widget) => widget.name === name)) return;
    if (kind === "lora") {
        const options = getOptions(nodeData, "lora_1");
        const selected = options.includes(value) ? value : (value ?? options[0] ?? "None");
        const widget = node.addWidget("combo", name, selected, null, { values: options });
        refreshModelManagerWidgets(node);
        return widget;
    }
    if (kind === "group") {
        const defaultValue = Number.isFinite(Number(value)) ? Number(value) : index;
        const inputData = ["INT", { default: defaultValue, min: 1, max: 10000, step: 1 }];
        if (ComfyWidgets?.INT) {
            const widget = ComfyWidgets.INT(node, name, inputData, app).widget;
            widget.value = defaultValue;
            return widget;
        }
        return node.addWidget("number", name, defaultValue, null, {
            min: 1,
            max: 10000,
            step: 10,
            precision: 0,
        });
    }
    return node.addWidget("number", name, value ?? 1.0, null, {
        min: -20,
        max: 20,
        step: 0.05,
        precision: 3,
    });
}

function normalizeLoraBaseWidgets(node, nodeData) {
    const options = nodeData ? getOptions(nodeData, "lora_1") : ["None"];
    const fallbackLora = options.includes("None") ? "None" : (options[0] ?? "None");
    for (const widget of node.widgets || []) {
        if (widget.name === "lora_1" && (widget.value == null || !options.includes(widget.value))) {
            widget.value = fallbackLora;
        } else if (widget.name === "strength_model_1" && !Number.isFinite(Number(widget.value))) {
            widget.value = 1.0;
        } else if (widget.name === "group_number_1" && !Number.isFinite(Number(widget.value))) {
            widget.value = 1;
        }
    }
}

function getLoraBaseValues(node) {
    const loraWidget = (node.widgets || []).find((widget) => widget.name === "lora_1");
    const strengthWidget = (node.widgets || []).find((widget) => widget.name === "strength_model_1");
    const groupWidget = (node.widgets || []).find((widget) => widget.name === "group_number_1");
    return {
        lora_1: loraWidget?.value,
        strength_model_1: strengthWidget?.value,
        group_number_1: groupWidget?.value,
    };
}

function restoreLoraBaseValues(node, nodeData, saved) {
    if (!saved || typeof saved !== "object") {
        normalizeLoraBaseWidgets(node, nodeData);
        return;
    }
    const options = getOptions(nodeData, "lora_1");
    for (const widget of node.widgets || []) {
        if (widget.name === "lora_1" && saved.lora_1 != null && options.includes(saved.lora_1)) {
            widget.value = saved.lora_1;
        } else if (widget.name === "strength_model_1" && Number.isFinite(Number(saved.strength_model_1))) {
            widget.value = Number(saved.strength_model_1);
        } else if (widget.name === "group_number_1" && Number.isFinite(Number(saved.group_number_1))) {
            widget.value = Number(saved.group_number_1);
        }
    }
    normalizeLoraBaseWidgets(node, nodeData);
}

function inferLoraBaseValuesFromWidgets(info, nodeData) {
    if (!Array.isArray(info?.widgets_values)) {
        return null;
    }
    const options = getOptions(nodeData, "lora_1");
    const lora = info.widgets_values.find((value) => typeof value === "string" && options.includes(value));
    const numbers = info.widgets_values.filter((value) => Number.isFinite(Number(value)));
    if (lora == null && !numbers.length) {
        return null;
    }
    return {
        lora_1: lora,
        strength_model_1: numbers[0],
        group_number_1: numbers[1] ?? 1,
    };
}

function removeLoraRow(node, index) {
    if (index === 1) {
        for (const widget of node.widgets || []) {
            if (widget.name === "lora_1") widget.value = "None";
            if (widget.name === "strength_model_1") widget.value = 0;
            if (widget.name === "group_number_1") widget.value = 1;
        }
        resizeNode(node);
        return;
    }
    for (const widget of [...(node.widgets || [])]) {
        const parsed = parseLoraWidgetName(widget.name);
        const shouldRemove = parsed?.index === index || widget.ognLoraRowRemove === index;
        if (!shouldRemove) continue;
        const widgetIndex = node.widgets.indexOf(widget);
        if (widgetIndex >= 0) node.widgets.splice(widgetIndex, 1);
    }
    orderLoraWidgets(node);
    resizeNode(node);
}

function addLoraRowRemoveButton(node, index) {
    for (const widget of [...(node.widgets || [])]) {
        if (widget.ognLoraRowRemove !== index) continue;
        const widgetIndex = node.widgets.indexOf(widget);
        if (widgetIndex >= 0) node.widgets.splice(widgetIndex, 1);
    }
    const widget = node.addWidget("button", "- Remove", null, () => removeLoraRow(node, index));
    widget.ognLoraRowRemove = index;
}

function addLoraRow(node, nodeData, index, values = {}) {
    addLoraValueWidget(node, nodeData, index, "lora", values.lora);
    addLoraValueWidget(node, nodeData, index, "strength", values.strength);
    addLoraValueWidget(node, nodeData, index, "group", values.group ?? values.group_number ?? index);
    addLoraRowRemoveButton(node, index);
}

function orderLoraWidgets(node) {
    if (!node.widgets) return;
    const widgetByName = new Map();
    const removeButtons = new Map();
    const passthrough = [];

    for (const widget of node.widgets) {
        const parsed = parseLoraWidgetName(widget.name);
        if (parsed) {
            widgetByName.set(widget.name, widget);
        } else if (widget.ognLoraRowRemove) {
            removeButtons.set(widget.ognLoraRowRemove, widget);
        } else if (widget !== node.ognButtonSpacer && widget !== node.ognAddLoraButton && widget !== node.ognAddSetButton) {
            passthrough.push(widget);
        }
    }

    const ordered = [...passthrough];
    for (const index of loraRowIndexes(node)) {
        const lora = widgetByName.get(loraWidgetName("lora", index));
        const strength = widgetByName.get(loraWidgetName("strength", index));
        const group = widgetByName.get(loraWidgetName("group", index));
        const remove = removeButtons.get(index);
        if (lora) ordered.push(lora);
        if (strength) ordered.push(strength);
        if (group) ordered.push(group);
        if (remove) ordered.push(remove);
    }

    if (node.ognButtonSpacer) ordered.push(node.ognButtonSpacer);
    if (node.ognAddLoraButton) ordered.push(node.ognAddLoraButton);
    node.widgets = ordered;
}

function moveLoraButtonsToEnd(node) {
    if (!node.widgets) return;
    if (node.ognAddLoraButton && !node.ognButtonSpacer) {
        node.ognButtonSpacer = {
            type: "custom",
            name: "ogn_button_spacer",
            value: null,
            serialize: false,
            computeSize: () => [0, 10],
            draw: () => { },
        };
    }
    for (const widget of [node.ognButtonSpacer, node.ognAddLoraButton]) {
        if (!widget) continue;
        const index = node.widgets.indexOf(widget);
        if (index >= 0) node.widgets.splice(index, 1);
        node.widgets.push(widget);
    }
    orderLoraWidgets(node);
}

function removeLegacyLoraSetWidgets(node) {
    for (const widget of [...(node.widgets || [])]) {
        if (widget === node.ognAddSetButton || widget.ognLoraHeaderSet || widget.ognLoraSetSpacer || widget.ognLoraSetAdd) {
            const index = node.widgets.indexOf(widget);
            if (index >= 0) node.widgets.splice(index, 1);
        }
    }
    node.ognAddSetButton = null;
}

function ensureLoraButtons(node, nodeData) {
    node.serialize_widgets = true;
    removeLegacyLoraSetWidgets(node);
    normalizeLoraBaseWidgets(node, nodeData);
    if (!node.ognAddLoraButton) {
        node.ognAddLoraButton = node.addWidget("button", "+ Add LoRA", null, () => {
            const index = nextLoraRowIndex(node);
            addLoraRow(node, nodeData, index, { group: index });
            moveLoraButtonsToEnd(node);
            resizeNode(node);
        });
    }
    for (const index of loraRowIndexes(node)) {
        addLoraRowRemoveButton(node, index);
    }
    normalizeLoraBaseWidgets(node, nodeData);
    moveLoraButtonsToEnd(node);
}

function saveLoraDynamicWidgets(node, nodeData) {
    normalizeLoraBaseWidgets(node, nodeData);
    return (node.widgets || [])
        .filter((widget) => {
            const parsed = parseLoraWidgetName(widget.name);
            return parsed && parsed.index !== 1;
        })
        .map((widget) => ({ name: widget.name, value: widget.value }));
}

function restoreLoraDynamicWidgets(node, nodeData, saved) {
    if (!Array.isArray(saved) || !saved.length) return;
    const values = new Map();
    for (const item of saved) {
        const parsed = parseLoraWidgetName(item.name);
        if (!parsed || parsed.index === 1) continue;
        const row = values.get(parsed.index) ?? { index: parsed.index, group: parsed.group };
        row[parsed.kind] = item.value;
        values.set(parsed.index, row);
    }
    for (const row of [...values.values()].sort((a, b) => a.index - b.index)) {
        addLoraRow(node, nodeData, row.index, row);
    }
    normalizeLoraBaseWidgets(node, nodeData);
    moveLoraButtonsToEnd(node);
    resizeNode(node);
}

function findWidgetByName(node, name) {
    return (node.widgets || []).find((widget) => widget.name === name);
}

function stripSafetensorsExt(name) {
    const text = String(name ?? "").replaceAll("\\", "/");
    return text.toLowerCase().endsWith(".safetensors") ? text.slice(0, -12) : text;
}

function ensureSafetensorsExt(name) {
    const text = String(name ?? "").trim().replaceAll("\\", "/");
    if (!text) return text;
    return text.toLowerCase().endsWith(".safetensors") ? text : `${text}.safetensors`;
}

function parseEpochLoraFile(name) {
    const text = String(name ?? "").trim().replaceAll("\\", "/");
    const stem = stripSafetensorsExt(text);
    const slash = stem.lastIndexOf("/");
    const dir = slash >= 0 ? stem.slice(0, slash + 1) : "";
    const fileStem = slash >= 0 ? stem.slice(slash + 1) : stem;
    const match = /^(.+)-0*([1-9]\d*)$/.exec(fileStem);
    if (!match) return null;
    return {
        base: `${dir}${match[1]}.safetensors`,
        epoch: Number(match[2]),
    };
}

function filenameOnly(path) {
    const text = String(path ?? "").replaceAll("\\", "/");
    return text.split("/").pop() ?? text;
}

function currentEpochRangeBasePath(node) {
    const selected = ensureSafetensorsExt(findWidgetByName(node, "lora_file")?.value);
    if (!selected) return "";

    if (node.ognEpochRangeUseSelectedAsBase === false) {
        const parsed = parseEpochLoraFile(selected);
        return parsed?.base ?? selected;
    }

    return selected;
}

function formatEpochRangeBaseDisplay(node, basePath) {
    const showRelative = Boolean(findWidgetByName(node, "show_relative_path")?.value);
    return showRelative ? basePath : filenameOnly(basePath);
}

function setWidgetValue(widget, value) {
    if (!widget) return;
    widget.value = value;
    widget.callback?.(value);
}

function updateLoraEpochRangeBaseDisplay(node) {
    const baseWidget = findWidgetByName(node, "base_lora_name");
    if (!baseWidget) return;

    const basePath = currentEpochRangeBasePath(node);
    baseWidget.value = formatEpochRangeBaseDisplay(node, basePath);
}

function markEpochRangeBaseWidget(node) {
    const baseWidget = findWidgetByName(node, "base_lora_name");
    if (!baseWidget) return;

    baseWidget.label = "base_lora_name";
    baseWidget.tooltip =
        "Display only. Toggle show_relative_path to switch between filename only and relative path.";
    if (baseWidget.inputEl) {
        baseWidget.inputEl.readOnly = true;
        baseWidget.inputEl.classList?.add("readonly");
    }
}

function moveEpochRangeWidgets(node) {
    if (!node.widgets) return;

    const button = node.ognInsertLastEpochButton;
    if (button && node.widgets.includes(button)) {
        const currentIndex = node.widgets.indexOf(button);
        if (currentIndex >= 0) node.widgets.splice(currentIndex, 1);

        const loraIndex = node.widgets.findIndex((widget) => widget.name === "lora_file");
        const insertIndex = loraIndex >= 0 ? loraIndex + 1 : node.widgets.length;
        node.widgets.splice(insertIndex, 0, button);
    }

    const showPathWidget = findWidgetByName(node, "show_relative_path");
    if (showPathWidget && node.widgets.includes(showPathWidget)) {
        const currentIndex = node.widgets.indexOf(showPathWidget);
        if (currentIndex >= 0) node.widgets.splice(currentIndex, 1);
        node.widgets.push(showPathWidget);
    }

    const baseWidget = findWidgetByName(node, "base_lora_name");
    if (baseWidget && node.widgets.includes(baseWidget)) {
        const currentIndex = node.widgets.indexOf(baseWidget);
        if (currentIndex >= 0) node.widgets.splice(currentIndex, 1);
        node.widgets.push(baseWidget);
    }
}

function ensureLoraEpochRangeButton(node, nodeData) {
    node.serialize_widgets = true;
    markEpochRangeBaseWidget(node);

    const loraWidget = findWidgetByName(node, "lora_file");
    if (loraWidget && !loraWidget.ognEpochRangeWrapped) {
        const originalCallback = loraWidget.callback;
        loraWidget.callback = function (value) {
            originalCallback?.apply(this, arguments);
            node.ognEpochRangeUseSelectedAsBase = true;

            const useDetectedWidget = findWidgetByName(node, "use_detected_epoch");
            setWidgetValue(useDetectedWidget, false);

            updateLoraEpochRangeBaseDisplay(node);
            resizeNode(node);
        };
        loraWidget.ognEpochRangeWrapped = true;
    }

    const showPathWidget = findWidgetByName(node, "show_relative_path");
    if (showPathWidget && !showPathWidget.ognEpochRangeWrapped) {
        const originalCallback = showPathWidget.callback;
        showPathWidget.callback = function (value) {
            originalCallback?.apply(this, arguments);
            updateLoraEpochRangeBaseDisplay(node);
            resizeNode(node);
        };
        showPathWidget.ognEpochRangeWrapped = true;
    }

    if (!node.ognInsertLastEpochButton) {
        node.ognInsertLastEpochButton = node.addWidget(
            "button",
            "Insert from last epoch file",
            null,
            () => {
                const selected = findWidgetByName(node, "lora_file")?.value;
                const parsed = parseEpochLoraFile(selected);

                if (!parsed) {
                    alert("File name must look like: foo-000120.safetensors");
                    return;
                }

                const lastEpochWidget = findWidgetByName(node, "last_epoch");
                const useDetectedWidget = findWidgetByName(node, "use_detected_epoch");

                node.ognEpochRangeUseSelectedAsBase = false;
                setWidgetValue(useDetectedWidget, true);
                setWidgetValue(lastEpochWidget, parsed.epoch);
                updateLoraEpochRangeBaseDisplay(node);
                resizeNode(node);
            }
        );
    }

    if (node.ognEpochRangeUseSelectedAsBase == null) {
        node.ognEpochRangeUseSelectedAsBase = true;
    }

    updateLoraEpochRangeBaseDisplay(node);
    moveEpochRangeWidgets(node);
    resizeNode(node);
}

function ensureButton(node, config, nodeData) {
    node.serialize_widgets = true;
    if (node.ognAddButton && (node.ognRemoveButton || !config.removeButton)) {
        moveButtonToEnd(node);
        return;
    }
    if (!node.ognAddButton) {
        node.ognAddButton = node.addWidget("button", config.button, null, () => {
            addRow(node, config, nodeData);
        });
    }
    if (config.removeButton && !node.ognRemoveButton) {
        node.ognRemoveButton = node.addWidget("button", config.removeButton, null, () => {
            removeLastRow(node, config);
        });
    }
    moveButtonToEnd(node);
}

function requestNodeRedraw() {
    app.graph?.setDirtyCanvas?.(true, false);
}

const CELL_PROGRESS_SLOT_START_Y = 40;
const CELL_PROGRESS_BAR_X = 18;
const CELL_PROGRESS_BAR_Y = 18;
const CELL_PROGRESS_BAR_HEIGHT = 16;

function drawCellProgress(node, ctx) {
    const progress = node.ognCellProgress;
    if (node.flags?.collapsed) return;

    const width = Math.max(0, node.size[0] - CELL_PROGRESS_BAR_X * 2);
    const barX = CELL_PROGRESS_BAR_X;
    const barY = CELL_PROGRESS_BAR_Y;
    const barH = CELL_PROGRESS_BAR_HEIGHT;
    const value = Number(progress?.value) || 0;
    const max = Number(progress?.max) || 0;
    const ratio = max > 0 ? Math.min(1, Math.max(0, value / max)) : 0;

    ctx.save();
    ctx.beginPath();
    ctx.roundRect(barX, barY, width, barH, 5);
    ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.18)";
    ctx.stroke();

    if (ratio > 0) {
        ctx.beginPath();
        ctx.roundRect(barX, barY, width * ratio, barH, 5);
        ctx.fillStyle = "#4b8ecf";
        ctx.fill();
    }

    ctx.fillStyle = max > 0 ? "#f2f2f2" : "rgba(242, 242, 242, 0.62)";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(max > 0 ? `Cells: ${value}/${max}` : "Cells: -/-", barX + width / 2, barY + barH / 2);
    ctx.restore();
}

function removeCellProgressWidget(node) {
    if (!node.widgets || !node.ognCellProgressWidget) return;

    const index = node.widgets.indexOf(node.ognCellProgressWidget);
    if (index >= 0) {
        node.widgets.splice(index, 1);
    }
    delete node.ognCellProgressWidget;
}

function ensureCellProgressLayout(node) {
    removeCellProgressWidget(node);
    node.constructor.slot_start_y = Math.max(node.constructor.slot_start_y ?? 0, CELL_PROGRESS_SLOT_START_Y);
    queueNodeResize(node);
}

function scheduleCellProgressWidget(node) {
    requestAnimationFrame(() => ensureCellProgressLayout(node));
}

function removeSerializedCellProgressValue(info) {
    if (Array.isArray(info?.widgets_values) && info.widgets_values[0] == null) {
        info.widgets_values.shift();
    }
}

function setWidgetVisible(node, widget, visible) {
    if (!widget) return;
    if (!widget.ognVisibilityState) {
        widget.ognVisibilityState = {
            type: widget.type,
            computeSize: widget.computeSize,
        };
    }
    widget.type = visible ? widget.ognVisibilityState.type : "ogn_hidden";
    widget.computeSize = visible ? widget.ognVisibilityState.computeSize : () => [0, -4];
}

function updateCellSaveWidgets(node) {
    const saveWidget = findWidgetByName(node, "save_each_cell_image");
    const directoryWidget = findWidgetByName(node, "cell_image_output_directory");
    const prefixWidget = findWidgetByName(node, "cell_image_filename_prefix");
    const loraNameWidget = findWidgetByName(node, "add_lora_name_to_cell_filename");
    if (!saveWidget || !directoryWidget || !prefixWidget || !loraNameWidget) return;

    saveWidget.tooltip = "Save each generated cell image separately. The current cell LoRA name is added to its filename.";
    directoryWidget.tooltip = "Subdirectory inside the ComfyUI output folder for saved cell images. Time tokens such as [time(%Y-%m-%d)] are supported. Leave blank to save in output.";
    prefixWidget.tooltip = "Filename prefix for saved cell images. Cell index is added automatically.";
    loraNameWidget.tooltip = "Append the current cell LoRA name after the filename prefix and cell index.";
    setWidgetVisible(node, directoryWidget, Boolean(saveWidget.value));
    setWidgetVisible(node, prefixWidget, Boolean(saveWidget.value));
    setWidgetVisible(node, loraNameWidget, Boolean(saveWidget.value));
    queueNodeResize(node);
}

function ensureCellSaveWidgets(node) {
    const saveWidget = findWidgetByName(node, "save_each_cell_image");
    if (!saveWidget || saveWidget.ognCellSaveWrapped) {
        updateCellSaveWidgets(node);
        return;
    }

    const callback = saveWidget.callback;
    saveWidget.callback = function () {
        callback?.apply(this, arguments);
        updateCellSaveWidgets(node);
    };
    saveWidget.ognCellSaveWrapped = true;
    updateCellSaveWidgets(node);
    queueNodeResize(node);
}

api.addEventListener("ogn_xy_plot/cell_progress", ({ detail }) => {
    const node = app.graph?.getNodeById?.(Number(detail?.node_id));
    if (!node) return;
    node.ognCellProgress = {
        value: Number(detail.value) || 0,
        max: Number(detail.max) || 0,
    };
    requestNodeRedraw();
});

function cellImageOutput(node) {
    return (node.outputs || []).find((output) => output.name === "cell_image") ?? node.outputs?.[1];
}

function linkedCellImageNodes(node) {
    const links = cellImageOutput(node)?.links || [];
    return links
        .map((linkId) => app.graph?.links?.[linkId])
        .map((link) => app.graph?.getNodeById?.(link?.target_id))
        .filter(Boolean);
}

function cellPreviewUrl(image) {
    const params = new URLSearchParams({
        filename: image.filename ?? "",
        subfolder: image.subfolder ?? "",
        type: image.type ?? "temp",
    });
    return `/view?${params.toString()}${app.getPreviewFormatParam?.() ?? ""}`;
}

function showCellPreviews(node, previews) {
    if (!Array.isArray(previews) || !previews.length) return;

    node.imgs = previews.map((preview) => {
        const image = new Image();
        image.onload = () => {
            node.setSizeForImage?.();
            node.setDirtyCanvas?.(true, true);
            requestNodeRedraw();
        };
        image.src = cellPreviewUrl(preview);
        return image;
    });
    node.setSizeForImage?.();
    node.setDirtyCanvas?.(true, true);
    requestNodeRedraw();
}

api.addEventListener("ogn_xy_plot/cell_preview", ({ detail }) => {
    const sourceNode = app.graph?.getNodeById?.(Number(detail?.node_id));
    if (!sourceNode) return;

    for (const targetNode of linkedCellImageNodes(sourceNode)) {
        showCellPreviews(
            targetNode,
            detail.images,
        );
    }
});

app.registerExtension({
    name: "ogn.xy_plot.dynamic_axes",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = NODE_CONFIG[nodeData.name];
        if (!config) {
            return;
        }
        if (config.cellProgress) {
            nodeType.slot_start_y = Math.max(nodeType.slot_start_y ?? 0, CELL_PROGRESS_SLOT_START_Y);

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                scheduleCellProgressWidget(this);
                ensureCellSaveWidgets(this);
            };

            const configure = nodeType.prototype.configure;
            nodeType.prototype.configure = function (info) {
                removeSerializedCellProgressValue(info);
                configure?.apply(this, arguments);
                ensureCellProgressLayout(this);
                ensureCellSaveWidgets(this);
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                onDrawForeground?.apply(this, arguments);
                drawCellProgress(this, ctx);
            };
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            if (config.loraSets) {
                ensureLoraButtons(this, nodeData);
                return;
            }
            if (config.loraEpochRange) {
                ensureLoraEpochRangeButton(this, nodeData);
                return;
            }
            ensureButton(this, config, nodeData);
        };

        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            configure?.apply(this, arguments);
            if (config.loraSets) {
                ensureLoraButtons(this, nodeData);
                restoreLoraBaseValues(this, nodeData, info?.ogn_lora_base ?? inferLoraBaseValuesFromWidgets(info, nodeData));
                restoreLoraDynamicWidgets(this, nodeData, info?.ogn_dynamic_widgets);
                return;
            }
            if (config.loraEpochRange) {
                ensureLoraEpochRangeButton(this, nodeData);
                return;
            }
            ensureButton(this, config, nodeData);
            restoreDynamicWidgets(this, config, nodeData, info?.ogn_dynamic_widgets);
        };

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
            onSerialize?.apply(this, arguments);
            if (config.loraSets) {
                normalizeLoraBaseWidgets(this, nodeData);
                data.ogn_lora_base = getLoraBaseValues(this);
                data.ogn_dynamic_widgets = saveLoraDynamicWidgets(this, nodeData);
                return;
            }
            if (config.loraEpochRange) {
                updateLoraEpochRangeBaseDisplay(this);
                return;
            }
            data.ogn_dynamic_widgets = saveDynamicWidgets(this, config);
        };
    },
});
