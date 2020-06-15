/*
 * File: app/store/ContributeWorkspacesCombo.js
 *
 * This file was generated by Sencha Architect version 3.1.0.
 * http://www.sencha.com/products/architect/
 *
 * This file requires use of the Ext JS 4.2.x library, under independent license.
 * License of Sencha Architect does not include license for Ext JS 4.2.x. For more
 * details see http://www.sencha.com/license or contact license@sencha.com.
 *
 * This file will be auto-generated each and everytime you save your project.
 *
 * Do NOT hand edit this file.
 */

Ext.define('Rubedo.store.ContributeWorkspacesCombo', {
    extend: 'Ext.data.Store',
    alias: 'store.ContributeWorkspacesCombo',

    requires: [
        'Rubedo.model.workspaceModel',
        'Ext.data.proxy.Ajax',
        'Ext.data.reader.Json'
    ],

    constructor: function(cfg) {
        var me = this;
        cfg = cfg || {};
        me.callParent([Ext.apply({
            isOptimised: true,
            usedCollection: 'Workspaces',
            autoLoad: true,
            model: 'Rubedo.model.workspaceModel',
            storeId: 'ContributeWorkspacesCombo',
            pageSize: 1000,
            proxy: {
                type: 'ajax',
                api: {
                    read: 'workspaces'
                },
                extraParams: {
                    notAll: true
                },
                reader: {
                    type: 'json',
                    messageProperty: 'message',
                    root: 'data'
                }
            },
            listeners: {
                load: {
                    fn: me.onJsonstoreLoad,
                    scope: me
                },
                beforeload: {
                    fn: me.onJsonstoreBeforeLoad,
                    scope: me
                }
            }
        }, cfg)]);
    },

    onJsonstoreLoad: function(store, records, successful, eOpts) {
        store.filter("canContribute", true);
    },

    onJsonstoreBeforeLoad: function(store, operation, eOpts) {
        store.clearFilter(true);
        store.getProxy().extraParams.filter=null;
    }

});