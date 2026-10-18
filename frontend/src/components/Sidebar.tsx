"use client";

import { useEffect, useMemo, useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import {
  Activity, AudioLines, BarChart3, BookOpen, Bug, Bot, Camera, Calendar, CalendarCheck, CalendarClock,
  ChartNoAxesColumn, ChartNoAxesCombined, CheckCircle2, ChevronDown, ChevronRight, CircleCheckBig, CircleX,
  CircleHelp, CloudDownload, CloudUpload, Clock3, Code2, Compass, ContactRound, Crosshair, Database, DollarSign,
  FileBarChart2, FileEdit, FilePenLine, FileSpreadsheet, FileText, Filter, Gavel, GitBranch, Globe2, GraduationCap,
  HeartPulse, Home, Inbox, Laptop, Library, LifeBuoy, Mail, Megaphone, MessageCircle, MessageSquareText, Mic, Monitor,
  MousePointerClick, Music2, Newspaper, Palette, PanelTop, PenLine, Plane, Plug, ReceiptText, Repeat2, Rocket,
  Search, SearchCheck, ShieldAlert, ShieldCheck, ShoppingBag, Sparkles, SquarePen, TestTube2, TriangleAlert, Trash2,
  UsersRound, Video, Webhook, Workflow, Wrench, X, Zap, Layers3,
} from "lucide-react";

interface NavItem { icon: React.ReactNode; label: string; href: string; badge?: string }

interface RecentTask {
  id: string;
  title: string;
  status?: "queued" | "running" | "paused" | "waiting_for_approval" | "completed" | "partial" | "failed" | "cancelled";
  domain?: string;
  updatedAt?: string;
  turnCount?: number;
  iconKey?: string;
  iconColor?: string;
  category?: string;
}

interface SidebarProps {
  onNewTask?: () => void;
  recentTasks?: RecentTask[];
  recentTasksLoading?: boolean;
  recentTasksError?: string | null;
  onRetryRecentTasks?: () => void;
  activeTaskId?: string | null;
  onSelectTask?: (id: string) => void;
  onDeleteTask?: (id: string) => void;
}

const TASK_COLOR_TOKENS: Record<string, { tile: string; icon: string }> = {
  indigo: { tile: "bg-[#f1f1ee]", icon: "text-[#34322d]" },
  violet: { tile: "bg-[#f1f1ee]", icon: "text-[#34322d]" },
  sky: { tile: "bg-[#f1f1ee]", icon: "text-[#34322d]" },
  emerald: { tile: "bg-[#f1f1ee]", icon: "text-[#34322d]" },
  cyan: { tile: "bg-[#f1f1ee]", icon: "text-[#34322d]" },
  pink: { tile: "bg-[#f1f1ee]", icon: "text-[#34322d]" },
  amber: { tile: "bg-[#f1f1ee]", icon: "text-[#34322d]" },
  slate: { tile: "bg-[#f1f1ee]", icon: "text-[#34322d]" },
};

const NAV_ITEMS: NavItem[] = [
  { icon: <Bot className="h-4 w-4" />, label: "Agent", href: "/", badge: "Live" },
  { icon: <Layers3 className="h-4 w-4" />, label: "Workspace", href: "/workspace" },
  { icon: <Calendar className="h-4 w-4" />, label: "Automations", href: "/automations" },
  { icon: <UsersRound className="h-4 w-4" />, label: "Command Center", href: "/command-center" },
  { icon: <Library className="h-4 w-4" />, label: "Source Library", href: "/library" },
  { icon: <Plug className="h-4 w-4" />, label: "Connectors", href: "/connectors" },
];
const SECONDARY_ITEMS: NavItem[] = [
  { icon: <Globe2 className="h-4 w-4" />, label: "Onsite48 Bridge", href: "/bridge" },
  { icon: <Clock3 className="h-4 w-4" />, label: "History", href: "/history" },
];

const ICONS: Record<string, { icon: typeof Search; color: string }> = {
  research: { icon: SearchCheck, color: "text-violet-600" }, search: { icon: Search, color: "text-violet-600" },
  investigation: { icon: Compass, color: "text-violet-600" }, knowledge: { icon: BookOpen, color: "text-indigo-600" },
  development: { icon: Code2, color: "text-emerald-600" }, code: { icon: Code2, color: "text-emerald-600" },
  coding: { icon: Code2, color: "text-emerald-600" }, repository: { icon: GitBranch, color: "text-orange-600" },
  bug: { icon: Bug, color: "text-red-600" }, testing: { icon: TestTube2, color: "text-emerald-600" },
  api: { icon: Webhook, color: "text-blue-600" }, deployment: { icon: Rocket, color: "text-orange-600" },
  email: { icon: Mail, color: "text-blue-600" },    communication: { icon: Mail, color: "text-blue-600" }, mail: { icon: Mail, color: "text-blue-600" },
  drafts: { icon: FilePenLine, color: "text-amber-600" }, inbox: { icon: Inbox, color: "text-blue-600" },
  browser: { icon: Globe2, color: "text-sky-600" }, website: { icon: Monitor, color: "text-sky-600" },
  globe: { icon: Globe2, color: "text-sky-600" }, active_tab: { icon: PanelTop, color: "text-blue-600" },
  click: { icon: MousePointerClick, color: "text-indigo-600" }, screenshot: { icon: Camera, color: "text-slate-600" },
  paired_device: { icon: Laptop, color: "text-emerald-600" }, approval_required: { icon: ShieldAlert, color: "text-amber-600" },
  marketing: { icon: Megaphone, color: "text-pink-600" }, chart: { icon: ChartNoAxesColumn, color: "text-pink-600" },
  analytics: { icon: ChartNoAxesCombined, color: "text-purple-600" }, funnel: { icon: Filter, color: "text-pink-600" },
  leads: { icon: ContactRound, color: "text-rose-600" },
  document: { icon: FileText, color: "text-slate-600" }, file: { icon: FileText, color: "text-slate-600" },
  writing: { icon: PenLine, color: "text-amber-600" }, editing: { icon: FileEdit, color: "text-orange-600" },
  report: { icon: FileBarChart2, color: "text-indigo-600" }, spreadsheet: { icon: FileSpreadsheet, color: "text-emerald-600" },
  data: { icon: Database, color: "text-cyan-600" }, database: { icon: Database, color: "text-cyan-600" },
  people: { icon: ContactRound, color: "text-rose-600" }, users: { icon: ContactRound, color: "text-rose-600" },
  work: { icon: Wrench, color: "text-orange-600" }, finance: { icon: DollarSign, color: "text-green-600" },
  invoice: { icon: ReceiptText, color: "text-amber-600" }, legal: { icon: Gavel, color: "text-red-600" },
   automation: { icon: Zap, color: "text-amber-600" }, workflow: { icon: Workflow, color: "text-orange-600" }, zap: { icon: Zap, color: "text-amber-600" },
  calendar: { icon: Calendar, color: "text-amber-600" }, meeting: { icon: CalendarCheck, color: "text-blue-500" },
  schedule: { icon: CalendarClock, color: "text-blue-600" }, recurring: { icon: Repeat2, color: "text-violet-600" },
  shopping: { icon: ShoppingBag, color: "text-fuchsia-600" }, travel: { icon: Plane, color: "text-sky-600" },
  health: { icon: HeartPulse, color: "text-red-600" }, education: { icon: GraduationCap, color: "text-violet-600" },
  home: { icon: Home, color: "text-teal-600" }, security: { icon: ShieldCheck, color: "text-green-600" },
  design: { icon: Palette, color: "text-fuchsia-600" }, image: { icon: Camera, color: "text-violet-600" },
  video: { icon: Video, color: "text-rose-600" }, voice: { icon: Mic, color: "text-blue-600" },
  music: { icon: Music2, color: "text-purple-600" }, audio: { icon: AudioLines, color: "text-indigo-600" },
  upload: { icon: CloudUpload, color: "text-cyan-600" }, download: { icon: CloudDownload, color: "text-cyan-600" },
  chat: { icon: MessageCircle, color: "text-indigo-600" }, support: { icon: LifeBuoy, color: "text-sky-500" },
  news: { icon: Newspaper, color: "text-slate-600" }, sparkles: { icon: Sparkles, color: "text-indigo-600" },
  danger: { icon: TriangleAlert, color: "text-red-600" }, weapons: { icon: Crosshair, color: "text-red-700" },
  violent_content: { icon: ShieldAlert, color: "text-red-600" }, hazard: { icon: TriangleAlert, color: "text-orange-600" },
  completed: { icon: CircleCheckBig, color: "text-emerald-600" }, failed: { icon: CircleX, color: "text-red-600" },
};

// Case-insensitive lookup so backend-sent keys like "Research", "WEB_SEARCH",
// or "Web Search" still match the ICONS record.
const ICONS_BY_KEY: Record<string, { icon: typeof Search; color: string }> = {};
Object.entries(ICONS).forEach(([key, value]) => {
  ICONS_BY_KEY[key.toLowerCase()] = value;
});

// Common backend value aliases -> canonical ICONS key.
const ICON_ALIASES: Record<string, string> = {
  web_search: "search", internet_search: "search", websearch: "search",
  google_search: "search", brave_search: "search", tavily: "search",
  research_task: "research", research: "research", market_research: "research",
  competitor_research: "research", code_dev: "development", coding: "development",
  software_dev: "development", dev: "development", web_dev: "development",
  email_task: "email", mail_task: "email", gmail: "email", outlook: "email",
  calendar_task: "calendar", scheduling: "calendar", meeting: "meeting",
  document_task: "document", doc: "document", docx: "document",
  data_task: "data", dataset: "data", spreadsheet_task: "spreadsheet",
  excel: "spreadsheet", csv: "spreadsheet", finance_task: "finance",
  accounting: "finance", budget: "finance", legal_task: "legal",
  compliance: "legal", contract: "legal",
};

function resolveIconKey(rawKey: string | undefined): string {
  if (!rawKey) return "";
  const lowered = rawKey.toLowerCase().trim();
  // Direct alias match first.
  if (ICON_ALIASES[lowered]) return ICON_ALIASES[lowered];
  // Case-insensitive direct match.
  if (ICONS_BY_KEY[lowered]) return lowered;
  // Try replacing spaces/underscores/hyphens with nothing.
  const normalized = lowered.replace(/[\s_\-]+/g, "");
  if (ICONS_BY_KEY[normalized]) return normalized;
  if (ICON_ALIASES[normalized]) return ICON_ALIASES[normalized];
  return "";
}

function fallbackIcon(title: string): string {
  const value = title.toLowerCase();
  if (/(gun|guns|knife|knives|weapon|weapons|firearm|violence|violent|danger|hazard|threat)/.test(value)) return "danger";
  if (/(voice|speech|transcribe|record audio|microphone|mic)/.test(value)) return "voice";
  if (/(music|song|playlist|audio|sound)/.test(value)) return "music";
  if (/(upload|attach|import file|send file)/.test(value)) return "upload";
  if (/(download|export file|save file)/.test(value)) return "download";
  if (/(click|press|tap|select button)/.test(value)) return "click";
  if (/(website|web site|webpage|web page|landing page)/.test(value)) return "website";
  if (/(research|search|compare|market|competitor)/.test(value)) return "research";
  if (/(code|github|bug|api|deploy)/.test(value)) return "development";
  if (/(email|mail|inbox|gmail|draft|message)/.test(value)) return "email";
  if (/(browser|website|web|url|site|youtube|linkedin)/.test(value)) return "browser";
  if (/(marketing|funnel|analytics|revenue)/.test(value)) return "marketing";
  if (/(data|dataset|spreadsheet|csv|metrics)/.test(value)) return "data";
  if (/(report|document|proposal|write|presentation)/.test(value)) return "document";
  if (/(meeting|call|schedule|appointment|calendar)/.test(value)) return "calendar";
  if (/(shopping|buy|purchase|order|cart)/.test(value)) return "shopping";
  if (/(travel|trip|flight|hotel|vacation)/.test(value)) return "travel";
  if (/(health|fitness|medical|doctor|wellness)/.test(value)) return "health";
  if (/(education|school|learn|course|study)/.test(value)) return "education";
  if (/(finance|money|payment|invoice|budget|tax)/.test(value)) return "finance";
  if (/(legal|contract|law|policy|compliance)/.test(value)) return "legal";
  return "search";
}

function TaskGlyph({ task }: { task: RecentTask }) {
  const fallback = fallbackIcon(task.title);
  const resolvedKey =
    resolveIconKey(task.iconKey) ||
    resolveIconKey(task.category) ||
    fallback;
  const entry = ICONS_BY_KEY[resolvedKey] || ICONS_BY_KEY[fallback] || ICONS_BY_KEY.search;
  const colors = TASK_COLOR_TOKENS[task.iconColor || ""] || { tile: "bg-[#f1f1ee]", icon: "text-[#34322d]" };
  const Icon = entry.icon;
  const running = task.status === "running" || task.status === "queued";
  const waiting = task.status === "paused" || task.status === "waiting_for_approval";
  const failed = task.status === "failed" || task.status === "cancelled";
  const statusLabel = running ? "Running" : waiting ? "Waiting" : failed ? "Failed" : "Completed";
  return <span className={`relative flex h-7 w-7 shrink-0 items-center justify-center rounded-lg ring-1 ring-black/[0.06] ${colors.tile}`} title={`${task.category || resolvedKey} · ${statusLabel}`}><Icon className={`h-4 w-4 ${colors.icon}`} /><span aria-label={statusLabel} className={`absolute -bottom-0.5 -right-0.5 h-1.5 w-1.5 rounded-full ring-1 ring-white ${running ? "animate-pulse bg-[#34322d]" : waiting ? "bg-[#6f6b61]" : failed ? "bg-[#171613]" : "bg-[#9a978c]"}`} /></span>;
}

function groupLabel(updatedAt?: string): "Today" | "Yesterday" | "Earlier" {
  if (!updatedAt) return "Earlier";
  const date = new Date(updatedAt);
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const taskDay = new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  const diff = Math.round((today - taskDay) / 86_400_000);
  return diff <= 0 ? "Today" : diff === 1 ? "Yesterday" : "Earlier";
}

export default function Sidebar({ onNewTask, recentTasks = [], recentTasksLoading = false, recentTasksError = null, onRetryRecentTasks, activeTaskId = null, onSelectTask, onDeleteTask }: SidebarProps) {
  const pathname = usePathname();
  const [mounted, setMounted] = useState(false);
  const [showRecentTasks, setShowRecentTasks] = useState(true);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "running" | "waiting" | "completed" | "failed">("all");
  useEffect(() => setMounted(true), []);
  const isActive = (href: string) => href === "/" ? pathname === "/" : pathname === href || pathname?.startsWith(`${href}/`);
  const visibleTasks = useMemo(() => recentTasks.filter((task) => {
    const haystack = `${task.title} ${task.category || ""} ${task.domain || ""}`.toLowerCase();
    if (search.trim() && !haystack.includes(search.trim().toLowerCase())) return false;
    if (statusFilter === "running") return task.status === "running" || task.status === "queued";
    if (statusFilter === "waiting") return task.status === "paused" || task.status === "waiting_for_approval";
    if (statusFilter === "completed") return task.status === "completed";
    if (statusFilter === "failed") return task.status === "failed" || task.status === "cancelled";
    return true;
  }), [recentTasks, search, statusFilter]);
  const groups = useMemo(() => (['Today', 'Yesterday', 'Earlier'] as const).map((label) => ({ label, tasks: visibleTasks.filter((task) => groupLabel(task.updatedAt) === label) })).filter((group) => group.tasks.length), [visibleTasks]);

  if (!mounted) return <aside className="h-screen w-64 shrink-0 border-r border-gray-200 bg-white" />;
  return <aside className="sticky top-0 flex h-screen w-64 shrink-0 flex-col border-r border-gray-200 bg-white">
    <div className="flex items-center gap-2 px-4 py-4"><img src="/Onsite48-opt.png" alt="Onsite48" className="h-5 w-5 object-contain" /><span className="text-lg font-bold tracking-tight text-[#34322d]">onsite48AI</span><span className="ml-auto text-xs text-gray-400">beta</span></div>
    <div className="px-3 pb-3"><button onClick={onNewTask} className="flex w-full items-center gap-2 rounded-lg bg-[#34322d] px-3 py-2.5 text-sm font-medium text-white transition hover:bg-[#171613]"><SquarePen className="h-4 w-4" /> New task</button></div>
    <nav className="space-y-0.5 px-3">{NAV_ITEMS.map((item) => <Link key={item.href} href={item.href} className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${isActive(item.href) ? "bg-blue-50 font-medium text-blue-700" : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"}`}><span className={isActive(item.href) ? "text-blue-600" : "text-gray-400"}>{item.icon}</span><span className="flex-1">{item.label}</span>{item.badge && <span className="rounded-full bg-blue-100 px-1.5 py-0.5 text-[10px] font-medium text-blue-700">{item.badge}</span>}</Link>)}</nav>
    <div className="mx-3 my-3 border-t border-gray-200" />
    <div className="flex min-h-0 flex-1 flex-col px-3">
      <div className="flex items-center justify-between"><button onClick={() => setShowRecentTasks(!showRecentTasks)} className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium text-gray-500 hover:bg-gray-100">{showRecentTasks ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />} TASKS</button>{showRecentTasks && recentTasks.length > 0 && <span className="text-[10px] text-gray-400">{visibleTasks.length}</span>}</div>
      {showRecentTasks && <>
        <div className="mt-1 flex items-center gap-1 rounded-lg border border-gray-200 bg-gray-50 px-2 py-1.5"><Search className="h-3.5 w-3.5 shrink-0 text-gray-400" /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search tasks" aria-label="Search tasks" className="min-w-0 flex-1 bg-transparent text-xs outline-none" />{search && <button onClick={() => setSearch("")} aria-label="Clear task search"><X className="h-3 w-3 text-gray-400" /></button>}</div>
        <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as typeof statusFilter)} aria-label="Filter tasks" className="mt-1.5 w-full rounded-md border-0 bg-transparent px-2 py-1 text-[11px] text-gray-500 outline-none"><option value="all">All tasks</option><option value="running">Running</option><option value="waiting">Waiting</option><option value="completed">Completed</option><option value="failed">Failed</option></select>
        <div className="min-h-0 flex-1 overflow-y-auto pb-3">
          {recentTasksLoading ? <div className="space-y-2 px-2 py-3" role="status"><span className="sr-only">Loading recent tasks</span>{[1, 2, 3].map((item) => <div key={item} className="h-8 animate-pulse rounded-lg bg-gray-100" />)}</div> : recentTasksError ? <div className="mx-1 rounded-lg border border-red-200 bg-red-50 px-2.5 py-3 text-xs text-red-700" role="alert"><p>Recent tasks could not be loaded.</p><button type="button" onClick={onRetryRecentTasks} className="mt-1.5 font-medium underline">Retry</button></div> : !visibleTasks.length ? <p className="px-2 py-4 text-xs text-gray-400">{search ? "No matching tasks" : "No tasks yet"}</p> : groups.map((group) => <section key={group.label} className="mt-2"><h3 className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-gray-400">{group.label}</h3>{group.tasks.map((task) => { const canDelete = ["completed", "failed", "cancelled"].includes(task.status || ""); return <div key={task.id} className="group flex items-center rounded-lg"><button onClick={() => onSelectTask?.(task.id)} title={task.title} className={`flex min-w-0 flex-1 items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm transition ${activeTaskId === task.id ? "bg-blue-50 font-medium text-blue-700" : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"}`}><TaskGlyph task={task} /><span className="min-w-0 flex-1"><span className="block truncate">{task.title}</span><span className="mt-0.5 block truncate text-[10px] font-normal text-gray-400">{task.status === "running" || task.status === "queued" ? "Running" : task.status === "waiting_for_approval" || task.status === "paused" ? "Waiting for you" : task.category || task.domain || "General"}{(task.turnCount || 0) > 1 ? ` · ${task.turnCount} turns` : ""}</span></span></button>{canDelete && <button type="button" onClick={(event) => { event.stopPropagation(); onDeleteTask?.(task.id); }} aria-label={`Delete task: ${task.title}`} title="Delete task" className="mr-1 shrink-0 rounded-md p-1.5 text-gray-400 opacity-0 transition hover:bg-red-50 hover:text-red-600 focus:opacity-100 focus:outline-none group-hover:opacity-100 group-focus-within:opacity-100"><Trash2 className="h-3.5 w-3.5" /></button>}</div>; })}</section>)}
        </div>
      </>}
    </div>
    <div className="border-t border-gray-200 px-3 py-3"><nav className="space-y-0.5">{SECONDARY_ITEMS.map((item) => <Link key={item.href} href={item.href} className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${isActive(item.href) ? "bg-blue-50 font-medium text-blue-700" : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"}`}><span className={isActive(item.href) ? "text-blue-600" : "text-gray-400"}>{item.icon}</span><span>{item.label}</span></Link>)}</nav></div>
  </aside>;
}