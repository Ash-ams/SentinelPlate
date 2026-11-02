import React, { useState, useEffect, useCallback } from 'react';
import {
  Car,
  CreditCard,
  ParkingCircle,
  BarChart3,
  Clock,
  ArrowDownLeft,
  ArrowUpRight,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Shield,
  Layers,
  Send,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  CartesianGrid,
} from 'recharts';
import { simulateTollEntry, getTollLogs, getTollStats } from '../api/client';

export default function TollDashboard({ initialPlate }) {
  const [logs, setLogs] = useState([]);
  const [stats, setStats] = useState({
    totalEntries: 0,
    totalExits: 0,
    totalRevenue: 0,
    currentOccupancy: 0,
    maxCapacity: 50,
    revenueByType: {
      white_private: 0,
      yellow_commercial: 0,
      green_ev: 0,
    },
  });

  const [formPlate, setFormPlate] = useState(initialPlate || '');
  const [formType, setFormType] = useState('white_private');
  const [formEvent, setFormEvent] = useState('entry');
  const [formGate, setFormGate] = useState('Gate A1');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [toast, setToast] = useState(null);
  const [isLoadingLogs, setIsLoadingLogs] = useState(false);

  // Sync initialPlate if passed from pipeline recognition
  useEffect(() => {
    if (initialPlate) {
      setFormPlate(initialPlate);
    }
  }, [initialPlate]);

  // Fetch logs and stats
  const refreshData = useCallback(async () => {
    setIsLoadingLogs(true);
    try {
      const [fetchedLogs, fetchedStats] = await Promise.all([
        getTollLogs(),
        getTollStats(),
      ]);
      setLogs(Array.isArray(fetchedLogs) ? fetchedLogs : []);
      if (fetchedStats) setStats(fetchedStats);
    } catch (err) {
      console.error('Failed to refresh toll dashboard:', err);
    } finally {
      setIsLoadingLogs(false);
    }
  }, []);

  useEffect(() => {
    refreshData();
  }, [refreshData]);

  // Form submission handler
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formPlate.trim()) return;

    setIsSubmitting(true);
    setToast(null);

    const payload = {
      plate_number: formPlate.trim().toUpperCase(),
      plate_type: formType,
      event_type: formEvent,
      gate_id: formGate,
    };

    try {
      const res = await simulateTollEntry(payload);
      setToast({
        type: 'success',
        message:
          res.message ||
          `Vehicle ${payload.plate_number} ${payload.event_type.toUpperCase()} recorded. Barrier OPENED.`,
      });
      // Auto refresh logs and stats after submission
      await refreshData();
    } catch (err) {
      setToast({
        type: 'error',
        message: err.message || 'Failed to simulate toll transaction',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  // Occupancy percentage calculation
  const occupancyPercent = Math.min(
    100,
    Math.round((stats.currentOccupancy / (stats.maxCapacity || 50)) * 100)
  );

  // Revenue chart data
  const revenueChartData = [
    {
      name: 'Private',
      type: 'white_private',
      revenue: stats.revenueByType?.white_private || 0,
      color: '#e2e8f0',
    },
    {
      name: 'Commercial',
      type: 'yellow_commercial',
      revenue: stats.revenueByType?.yellow_commercial || 0,
      color: '#eab308',
    },
    {
      name: 'Electric (EV)',
      type: 'green_ev',
      revenue: stats.revenueByType?.green_ev || 0,
      color: '#10b981',
    },
  ];

  return (
    <div className="flex flex-col gap-6">
      {/* Toast / Alert Message */}
      {toast && (
        <div
          className={`px-4 py-3 rounded-xl border flex items-center justify-between gap-3 text-xs shadow-lg animate-in fade-in duration-200 ${
            toast.type === 'success'
              ? 'bg-emerald-950/80 border-emerald-800 text-emerald-200'
              : 'bg-rose-950/80 border-rose-800 text-rose-200'
          }`}
        >
          <div className="flex items-center gap-2">
            {toast.type === 'success' ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            )}
            <span className="font-medium">{toast.message}</span>
          </div>
          <button
            onClick={() => setToast(null)}
            className="text-gray-400 hover:text-white text-xs font-semibold px-2 py-0.5"
          >
            ✕
          </button>
        </div>
      )}

      {/* Top Row: Key Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* 1. Total Entries */}
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
              Total Entries
            </span>
            <h3 className="text-2xl font-bold font-mono text-white mt-1">
              {stats.totalEntries}
            </h3>
            <span className="text-[11px] text-indigo-400 flex items-center gap-1 mt-1">
              <ArrowDownLeft className="w-3 h-3" /> Inflow stream active
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
            <Car className="w-6 h-6" />
          </div>
        </div>

        {/* 2. Total Exits */}
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
              Total Exits
            </span>
            <h3 className="text-2xl font-bold font-mono text-white mt-1">
              {stats.totalExits}
            </h3>
            <span className="text-[11px] text-emerald-400 flex items-center gap-1 mt-1">
              <ArrowUpRight className="w-3 h-3" /> Outflow barrier cleared
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <Clock className="w-6 h-6" />
          </div>
        </div>

        {/* 3. Total Toll Revenue */}
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
              Toll Revenue
            </span>
            <h3 className="text-2xl font-bold font-mono text-amber-300 mt-1">
              ₹{stats.totalRevenue.toLocaleString()}
            </h3>
            <span className="text-[11px] text-gray-400 flex items-center gap-1 mt-1">
              Fastag & automated tariff
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
            <CreditCard className="w-6 h-6" />
          </div>
        </div>

        {/* 4. Occupancy with progress bar */}
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
              Lot Occupancy
            </span>
            <ParkingCircle className="w-5 h-5 text-indigo-400" />
          </div>
          <div className="mt-1">
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-bold font-mono text-white">
                {stats.currentOccupancy}
                <span className="text-sm font-normal text-gray-500">
                  /{stats.maxCapacity || 50} Bays
                </span>
              </span>
              <span className="text-xs font-semibold text-indigo-400">
                {occupancyPercent}%
              </span>
            </div>
            {/* Progress bar */}
            <div className="w-full bg-gray-950 rounded-full h-2 mt-2 overflow-hidden border border-gray-800">
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  occupancyPercent > 85
                    ? 'bg-rose-500'
                    : occupancyPercent > 60
                    ? 'bg-amber-500'
                    : 'bg-indigo-500'
                }`}
                style={{ width: `${occupancyPercent}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Middle Row: Manual Simulation Form & Revenue Breakdown Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Manual Toll Simulation Form */}
        <div className="lg:col-span-2 bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl flex flex-col gap-4">
          <div className="flex items-center justify-between pb-3 border-b border-gray-800">
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-indigo-400" />
              <h2 className="text-sm font-semibold text-white">
                Simulate Toll Gate Event
              </h2>
            </div>
            <span className="text-[11px] text-gray-400">
              Simulates ALPR trigger at entrance or exit boom barrier
            </span>
          </div>

          <form onSubmit={handleSubmit} className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Plate Number */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-gray-300">
                License Plate Number
              </label>
              <input
                type="text"
                value={formPlate}
                onChange={(e) => setFormPlate(e.target.value.toUpperCase())}
                placeholder="e.g. MH46P1661"
                required
                className="w-full bg-gray-950 border border-gray-800 text-sm font-mono uppercase tracking-wider text-white rounded-xl px-3.5 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500"
              />
            </div>

            {/* Plate Type */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-gray-300">
                Plate Category / Tariff Type
              </label>
              <select
                value={formType}
                onChange={(e) => setFormType(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 text-xs text-gray-200 rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500"
              >
                <option value="white_private">White Plate (Private Car - ₹60)</option>
                <option value="yellow_commercial">Yellow Plate (Commercial / Taxi - ₹120)</option>
                <option value="green_ev">Green Plate (Electric Vehicle - ₹30)</option>
              </select>
            </div>

            {/* Event Type Radio */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-gray-300">
                Gate Event
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setFormEvent('entry')}
                  className={`py-2 px-3 rounded-xl text-xs font-medium flex items-center justify-center gap-1.5 border transition-all ${
                    formEvent === 'entry'
                      ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300 shadow-sm'
                      : 'bg-gray-950 border-gray-800 text-gray-400 hover:text-white'
                  }`}
                >
                  <ArrowDownLeft className="w-3.5 h-3.5" />
                  <span>Entry Gate</span>
                </button>
                <button
                  type="button"
                  onClick={() => setFormEvent('exit')}
                  className={`py-2 px-3 rounded-xl text-xs font-medium flex items-center justify-center gap-1.5 border transition-all ${
                    formEvent === 'exit'
                      ? 'bg-emerald-600/20 border-emerald-500 text-emerald-300 shadow-sm'
                      : 'bg-gray-950 border-gray-800 text-gray-400 hover:text-white'
                  }`}
                >
                  <ArrowUpRight className="w-3.5 h-3.5" />
                  <span>Exit Gate</span>
                </button>
              </div>
            </div>

            {/* Gate ID */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-gray-300">
                Toll Lane / Gate ID
              </label>
              <select
                value={formGate}
                onChange={(e) => setFormGate(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 text-xs text-gray-200 rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500"
              >
                <option value="Gate A1">Gate A1 (North Express Lane)</option>
                <option value="Gate A2">Gate A2 (North Regular Lane)</option>
                <option value="Gate B1">Gate B1 (South Fastag Lane)</option>
                <option value="Gate B2">Gate B2 (South Exit Lane)</option>
                <option value="Gate C1">Gate C1 (Commercial Heavy Lane)</option>
              </select>
            </div>

            {/* Submit Button */}
            <div className="sm:col-span-2 pt-1">
              <button
                type="submit"
                disabled={isSubmitting || !formPlate.trim()}
                className="w-full py-2.5 px-4 rounded-xl font-semibold text-xs flex items-center justify-center gap-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-md shadow-indigo-600/20 transition-all cursor-pointer disabled:opacity-50"
              >
                {isSubmitting ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-indigo-300" />
                    <span>Communicating with Barrier Controller...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-3.5 h-3.5" />
                    <span>Trigger Toll Simulation</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Revenue by Plate Type Chart */}
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl flex flex-col gap-3">
          <div className="flex items-center justify-between pb-2 border-b border-gray-800">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-amber-400" />
              <h2 className="text-xs font-semibold text-white uppercase tracking-wider">
                Revenue by Plate Type
              </h2>
            </div>
          </div>

          <div className="h-44 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={revenueChartData}
                margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis
                  dataKey="name"
                  stroke="#6b7280"
                  tick={{ fill: '#9ca3af', fontSize: 10 }}
                  tickLine={false}
                />
                <YAxis
                  stroke="#6b7280"
                  tick={{ fill: '#9ca3af', fontSize: 10 }}
                  tickLine={false}
                  tickFormatter={(v) => `₹${v}`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#111827',
                    borderColor: '#374151',
                    borderRadius: '0.5rem',
                    fontSize: '11px',
                    color: '#f3f4f6',
                  }}
                  formatter={(val) => [`₹${val}`, 'Toll Collected']}
                />
                <Bar dataKey="revenue" radius={[6, 6, 0, 0]}>
                  {revenueChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-3 gap-1 pt-1 border-t border-gray-800 text-center">
            <div>
              <span className="text-[10px] text-gray-500">Private</span>
              <p className="text-xs font-mono font-bold text-gray-200">
                ₹{stats.revenueByType?.white_private || 0}
              </p>
            </div>
            <div>
              <span className="text-[10px] text-amber-400/80">Commercial</span>
              <p className="text-xs font-mono font-bold text-amber-400">
                ₹{stats.revenueByType?.yellow_commercial || 0}
              </p>
            </div>
            <div>
              <span className="text-[10px] text-emerald-400/80">EV</span>
              <p className="text-xs font-mono font-bold text-emerald-400">
                ₹{stats.revenueByType?.green_ev || 0}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Section: Toll Audit Logs Table */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl flex flex-col gap-4">
        <div className="flex items-center justify-between pb-3 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-indigo-400" />
            <h2 className="text-sm font-semibold text-white">
              Toll & Parking Transaction Logs
            </h2>
          </div>
          <button
            onClick={refreshData}
            disabled={isLoadingLogs}
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-indigo-400 transition-colors disabled:opacity-50"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${isLoadingLogs ? 'animate-spin' : ''}`}
            />
            <span>Refresh Audit Trail</span>
          </button>
        </div>

        {/* Scrollable Table */}
        <div className="overflow-x-auto rounded-xl border border-gray-800">
          <table className="w-full text-left text-xs text-gray-300">
            <thead className="bg-gray-950 text-gray-400 font-semibold uppercase tracking-wider text-[10px] border-b border-gray-800">
              <tr>
                <th className="px-4 py-3">Log ID</th>
                <th className="px-4 py-3">Plate Number</th>
                <th className="px-4 py-3">Plate Type</th>
                <th className="px-4 py-3">Event</th>
                <th className="px-4 py-3">Lane / Gate</th>
                <th className="px-4 py-3">Timestamp</th>
                <th className="px-4 py-3">Assigned Bay</th>
                <th className="px-4 py-3">Duration</th>
                <th className="px-4 py-3">Tariff</th>
                <th className="px-4 py-3">Barrier</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/80 bg-gray-900/60 font-mono">
              {logs.length === 0 ? (
                <tr>
                  <td
                    colSpan="10"
                    className="px-4 py-8 text-center text-gray-500 font-sans text-xs"
                  >
                    No toll records found. Trigger an event above.
                  </td>
                </tr>
              ) : (
                logs.map((log) => {
                  const isEntry = log.event === 'entry';
                  return (
                    <tr
                      key={log.id}
                      className="hover:bg-gray-850/80 transition-colors"
                    >
                      <td className="px-4 py-2.5 text-gray-400">{log.id}</td>
                      <td className="px-4 py-2.5">
                        <span className="font-bold text-white tracking-wider">
                          {log.plate}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 font-sans">
                        <span
                          className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${
                            log.plate_type === 'yellow_commercial'
                              ? 'bg-amber-400/10 text-amber-400 border border-amber-400/20'
                              : log.plate_type === 'green_ev'
                              ? 'bg-emerald-400/10 text-emerald-400 border border-emerald-400/20'
                              : 'bg-gray-700/30 text-gray-300 border border-gray-700'
                          }`}
                        >
                          {log.plate_type ? log.plate_type.replace('_', ' ') : 'standard'}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 font-sans">
                        <span
                          className={`inline-flex items-center gap-1 text-[11px] font-semibold px-2 py-0.5 rounded ${
                            isEntry
                              ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20'
                              : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          }`}
                        >
                          {isEntry ? (
                            <ArrowDownLeft className="w-3 h-3" />
                          ) : (
                            <ArrowUpRight className="w-3 h-3" />
                          )}
                          <span className="uppercase">{log.event}</span>
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-gray-300 font-sans">
                        {log.gate}
                      </td>
                      <td className="px-4 py-2.5 text-gray-400">{log.time}</td>
                      <td className="px-4 py-2.5 text-indigo-300 font-semibold">
                        {log.bay || 'N/A'}
                      </td>
                      <td className="px-4 py-2.5 text-gray-400">
                        {log.duration || '0m'}
                      </td>
                      <td className="px-4 py-2.5 font-bold text-amber-300">
                        ₹{log.tariff || 0}
                      </td>
                      <td className="px-4 py-2.5 font-sans">
                        <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                          {log.barrier || 'OPEN'}
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
