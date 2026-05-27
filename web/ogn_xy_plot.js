import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_CONFIG = {
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
      draw: () => {},
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
    node.addWidget("number", name, value ?? row.value ?? 1.0, null, {
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
    node.addWidget("text", name, value ?? row.value ?? "", null, {});
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
    draw: () => {},
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
  node.size[0] = Math.max(node.size[0], size[0]);
  node.size[1] = Math.max(node.size[1], size[1]);
  node.setDirtyCanvas(true, true);
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
    const shouldRemove =
      parsed?.index === index ||
      widget.ognLoraRowRemove === index ||
      widget.ognLoraRowSpacer === index;
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

function ensureLoraRowSpacer(node, index) {
  if ((node.widgets || []).some((widget) => widget.ognLoraRowSpacer === index)) return;
  node.widgets.push(createLoraRowSpacer(index));
}

function createLoraRowSpacer(index) {
  return {
    type: "custom",
    name: `ogn_lora_row_spacer_${index}`,
    value: null,
    serialize: false,
    ognLoraRowSpacer: index,
    computeSize: () => [0, 10],
    draw: () => {},
  };
}

function addLoraRow(node, nodeData, index, values = {}) {
  addLoraValueWidget(node, nodeData, index, "lora", values.lora);
  addLoraValueWidget(node, nodeData, index, "strength", values.strength);
  addLoraValueWidget(node, nodeData, index, "group", values.group ?? values.group_number ?? index);
  addLoraRowRemoveButton(node, index);
  ensureLoraRowSpacer(node, index);
}

function orderLoraWidgets(node) {
  if (!node.widgets) return;
  const widgetByName = new Map();
  const removeButtons = new Map();
  const spacers = new Map();
  const passthrough = [];

  for (const widget of node.widgets) {
    const parsed = parseLoraWidgetName(widget.name);
    if (parsed) {
      widgetByName.set(widget.name, widget);
    } else if (widget.ognLoraRowRemove) {
      removeButtons.set(widget.ognLoraRowRemove, widget);
    } else if (widget.ognLoraRowSpacer) {
      spacers.set(widget.ognLoraRowSpacer, widget);
    } else if (
      widget !== node.ognButtonSpacer &&
      widget !== node.ognAddLoraButton &&
      widget !== node.ognAddSetButton
    ) {
      passthrough.push(widget);
    }
  }

  const ordered = [...passthrough];
  const rowIndexes = loraRowIndexes(node);
  for (const [rowPosition, index] of rowIndexes.entries()) {
    const lora = widgetByName.get(loraWidgetName("lora", index));
    const strength = widgetByName.get(loraWidgetName("strength", index));
    const group = widgetByName.get(loraWidgetName("group", index));
    const remove = removeButtons.get(index);
    const spacer = spacers.get(index);
    if (lora) ordered.push(lora);
    if (strength) ordered.push(strength);
    if (group) ordered.push(group);
    if (remove) ordered.push(remove);
    if (rowPosition < rowIndexes.length - 1) ordered.push(spacer ?? createLoraRowSpacer(index));
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
      draw: () => {},
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
    if (
      widget === node.ognAddSetButton ||
      widget.ognLoraHeaderSet ||
      widget.ognLoraSetSpacer ||
      widget.ognLoraSetAdd
    ) {
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
    ensureLoraRowSpacer(node, index);
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

app.registerExtension({
  name: "ogn.xy_plot.dynamic_axes",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const config = NODE_CONFIG[nodeData.name];
    if (!config) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      if (config.loraSets) {
        ensureLoraButtons(this, nodeData);
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
      data.ogn_dynamic_widgets = saveDynamicWidgets(this, config);
    };
  },
});
